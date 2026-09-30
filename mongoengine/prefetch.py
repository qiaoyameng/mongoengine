"""Explicit, budgeted eager dereferencing (a bounded alternative to
``select_related``).

``select_related`` walks the whole reference graph: cyclic graphs are only cut
by ``max_depth`` and shared references can pull a huge fraction of the
database into memory.

This module provides :func:`prefetch_related`, which walks the graph in
rounds (one batched query per referenced collection per round) while enforcing
an explicit :class:`PrefetchBudget`:

* ``max_depth`` - how many reference hops away from the roots are resolved;
* ``max_objects`` - number of *unique* referenced documents pulled from the
  database (the same document reached through several paths only counts once);
* ``max_queries_per_round`` - number of batched ``find`` queries issued in a
  single round.

References that could not be resolved because a budget was exhausted are left
in place as :class:`~mongoengine.base.datastructures.UnresolvedReference`
markers. Markers are strictly lazy (printing/serialising them never issues a
query) and carry a dotted ``path`` explaining where in the graph they sit.

A :class:`PrefetchReport` is always returned, describing what was resolved and
which paths remain unresolved.
"""

import threading
from dataclasses import dataclass, field

from bson import DBRef

from mongoengine.base import (
    LazyReference,
    UnresolvedReference,
    _DocumentRegistry,
)
from mongoengine.connection import _get_session, get_db
from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (
    DictField,
    GenericReferenceField,
    ListField,
    ReferenceField,
)
from mongoengine.queryset import QuerySet

__all__ = (
    "PrefetchBudget",
    "PrefetchReport",
    "prefetch_related",
    "last_report",
    "UnresolvedReference",
)

# Reasons an edge ends up unresolved.
REASON_DEPTH = "depth"
REASON_OBJECTS = "objects"
REASON_QUERIES = "queries"
REASON_MISSING = "missing"
REASON_CLASS = "unregistered"

_last_report_store = threading.local()


def last_report():
    """Return the :class:`PrefetchReport` produced by the most recent
    budgeted prefetch run on this thread (also the one behind the last
    ``QuerySet.select_related`` call with a budget), or ``None``."""
    return getattr(_last_report_store, "report", None)


class PrefetchBudget:
    """Limits applied while eagerly dereferencing references.

    :param max_depth: Maximum number of reference hops to follow starting from
        the root objects. ``0`` means "do not dereference anything", ``1``
        resolves only direct references, and so on. ``None`` removes the depth
        limit (the walk is still bounded by the other budgets and by cycle
        detection).
    :param max_objects: Maximum number of *unique* referenced documents pulled
        from the database. A document referenced from several paths counts
        only once. ``None`` removes the limit.
    :param max_queries_per_round: Maximum number of batched queries issued in a
        single round (one query per referenced collection). Collections that do
        not fit into a round are retried in the next round. ``None`` removes
        the limit.
    """

    def __init__(
        self, max_depth=1, max_objects=None, max_queries_per_round=None
    ):
        self.max_depth = self._validate("max_depth", max_depth)
        self.max_objects = self._validate("max_objects", max_objects)
        self.max_queries_per_round = self._validate(
            "max_queries_per_round", max_queries_per_round
        )
        if self.max_queries_per_round == 0:
            raise ValueError(
                "max_queries_per_round must be a positive integer or None "
                "(use max_objects=0 to resolve nothing)"
            )

    @staticmethod
    def _validate(name, value):
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be a non-negative integer or None")
            if value < 0:
                raise ValueError(f"{name} must be a non-negative integer or None")
        return value

    def __repr__(self):
        return (
            "PrefetchBudget(max_depth=%r, max_objects=%r, "
            "max_queries_per_round=%r)"
            % (self.max_depth, self.max_objects, self.max_queries_per_round)
        )


@dataclass
class PrefetchReport:
    """Result of a budgeted prefetch run.

    :ivar budget: the :class:`PrefetchBudget` that was enforced.
    :ivar rounds: number of query rounds that were performed.
    :ivar query_count: total number of ``find`` queries issued.
    :ivar queries_by_round: number of queries issued in each round, in order.
    :ivar objects_fetched: number of unique referenced documents pulled from
        the database (root objects are not counted).
    :ivar resolved_paths: dotted paths of references that got resolved.
    :ivar unresolved: ``{path: reason}`` for references left as
        :class:`~mongoengine.base.datastructures.UnresolvedReference` markers.
        ``reason`` is one of ``"depth"``, ``"objects"``, ``"queries"``,
        ``"missing"`` or ``"unregistered"``.
    :ivar unresolved_paths: sorted tuple of the keys of ``unresolved``.
    :ivar incomplete: ``True`` when at least one reachable reference was not
        resolved because a budget was exhausted.
    """

    budget: object
    rounds: int = 0
    query_count: int = 0
    queries_by_round: list = field(default_factory=list)
    objects_fetched: int = 0
    resolved_paths: list = field(default_factory=list)
    unresolved: dict = field(default_factory=dict)

    @property
    def unresolved_paths(self):
        return tuple(sorted(self.unresolved))

    @property
    def incomplete(self):
        return any(
            reason in (REASON_DEPTH, REASON_OBJECTS, REASON_QUERIES)
            for reason in self.unresolved.values()
        )

    def describe_unresolved(self):
        """Return a human-readable description of the unresolved paths."""
        if not self.unresolved:
            return "all reachable references were resolved"
        lines = [
            f"{path} ({reason})"
            for path, reason in sorted(self.unresolved.items())
        ]
        return "unresolved paths:\n    " + "\n    ".join(lines)

    def __str__(self):
        status = "incomplete" if self.incomplete else "complete"
        return (
            f"<PrefetchReport {status}: rounds={self.rounds}, "
            f"queries={self.query_count}, objects={self.objects_fetched}, "
            f"unresolved={len(self.unresolved)}>"
        )

    def __repr__(self):
        return self.__str__()


def _unwrapped_field(field):
    """Strip ListField/DictField/MapField wrappers, returning the innard."""
    seen = set()
    while isinstance(field, (ListField, DictField)) and field.field is not None:
        if id(field) in seen:
            break
        seen.add(id(field))
        field = field.field
    return field


def _doc_identity(doc):
    return (type(doc)._get_collection_name(), doc.pk)


def _is_pending(value):
    """Value still stored as a raw reference (not a fetched document)."""
    if isinstance(value, UnresolvedReference):
        return False
    if isinstance(value, LazyReference):
        # Lazily-loaded fields are not part of eager prefetching.
        return False
    if isinstance(value, DBRef):
        return True
    if isinstance(value, dict) and "_ref" in value and "_cls" in value:
        return True
    return False


def _edge_from_value(value, document_type=None):
    """Describe a pending reference value.

    Returns ``(kind, key, document_type, object_id)`` where kind is
    ``"typed"``, ``"generic"`` or ``"collection"``, or ``None`` if the value
    is not a pending reference.
    """
    if isinstance(value, DBRef):
        if document_type is not None:
            return ("typed", document_type, document_type, value.id)
        return ("collection", value.collection, None, value.id)
    if isinstance(value, dict) and "_ref" in value and "_cls" in value:
        try:
            doc_cls = _DocumentRegistry.get(value["_cls"])
        except Exception:
            return ("collection", value["_ref"].collection, None, value["_ref"].id)
        # Once the generic reference resolves to a registered document class
        # it is fetched exactly like a typed reference, so both edge kinds
        # share a fetch group (one query per collection).
        return ("typed", doc_cls, doc_cls, value["_ref"].id)
    return None



def _iter_field_values(doc):
    fields = dict(doc._fields)
    if getattr(doc, "_dynamic", False):
        fields.update(doc._dynamic_fields)
    for field_name, field in fields.items():
        yield field_name, field, doc._data.get(field_name, None)


def _scan_container(value, field, path, slots, document_type=None):
    """Walk nested lists/dicts/tuples looking for pending references."""
    inner = _unwrapped_field(field) if field is not None else None
    if isinstance(inner, ReferenceField):
        document_type = inner.document_type

    iterator = value.items() if isinstance(value, dict) else enumerate(value)
    for key, item in iterator:
        item_path = f"{path}[{key!r}]" if isinstance(value, dict) else f"{path}[{key}]"
        if _is_pending(item):
            edge = _edge_from_value(item, document_type)
            if edge is not None:
                slots.append(_Slot(value, key, item_path, edge))
        elif isinstance(item, EmbeddedDocument):
            _scan_document(item, item_path, slots)
        elif isinstance(item, (list, tuple, dict)):
            _scan_container(item, inner, item_path, slots)
    return slots


def _scan_document(doc, path, slots):
    """Collect pending reference slots inside a (possibly embedded) document."""
    for field_name, field, value in _iter_field_values(doc):
        if value is None:
            continue
        field_path = f"{path}.{field_name}" if path else field_name
        inner = _unwrapped_field(field)

        if isinstance(inner, ReferenceField):
            if _is_pending(value):
                edge = _edge_from_value(value, inner.document_type)
                if edge is not None:
                    slots.append(_Slot(doc._data, field_name, field_path, edge))
            elif isinstance(value, (list, tuple, dict)):
                _scan_container(value, field, field_path, slots)
            elif isinstance(value, EmbeddedDocument):
                _scan_document(value, field_path, slots)
        elif isinstance(inner, GenericReferenceField):
            if _is_pending(value):
                edge = _edge_from_value(value)
                if edge is not None:
                    slots.append(_Slot(doc._data, field_name, field_path, edge))
            elif isinstance(value, (list, tuple, dict)):
                _scan_container(value, field, field_path, slots)
        elif isinstance(value, EmbeddedDocument):
            _scan_document(value, field_path, slots)
        elif isinstance(value, (list, tuple, dict)):
            _scan_container(value, field, field_path, slots)


class _Slot:
    """A location holding a pending reference inside the object graph."""

    __slots__ = ("container", "key", "path", "edge", "resolved")

    def __init__(self, container, key, path, edge):
        self.container = container
        self.key = key
        self.path = path
        self.edge = edge
        self.resolved = False

    def set(self, value):
        self.container[self.key] = value


def _fetch_group(doc_cls, ids):
    """Issue a single batched query for a set of ids of one document class."""
    return doc_cls.objects().in_bulk(list(ids))


def _fetch_raw_group(collection_name, ids):
    """Fallback for references whose document class could not be resolved."""
    cursor = get_db()[collection_name].find(
        {"_id": {"$in": list(ids)}}, session=_get_session()
    )
    docs = {}
    for son in cursor:
        cls_name = son.get("_cls")
        if cls_name:
            doc_cls = _DocumentRegistry.get(cls_name)
        else:
            doc_cls = _DocumentRegistry.get(
                "".join(part.capitalize() for part in collection_name.split("_"))
            )
        doc = doc_cls._from_son(son)
        docs[doc.pk] = doc
    return docs


def _collect_slots(docs, slots, doc_paths):
    """Scan frontier documents.

    ``doc_paths`` maps each document to the dotted path through which it was
    reached; this keeps report paths unique even when the same object is
    reached again while closing a cycle.
    """
    for doc in docs:
        if not isinstance(doc, Document):
            continue
        path = doc_paths.get(id(doc)) or "root"
        _scan_document(doc, path, slots)


def _group_slots(slots, object_map):
    """Attach slots targeting already-materialised objects; group the rest.

    Returns ``(groups, slot_targets)`` where ``groups`` maps
    ``(kind, target_key)`` to a set of ids and ``slot_targets`` maps
    ``(kind, target_key, id)`` to the list of slots pointing there.
    """
    groups = {}
    slot_targets = {}
    for slot in slots:
        kind, key, doc_cls, obj_id = slot.edge
        identity = (
            key._get_collection_name() if doc_cls is not None else key
        )
        map_key = (identity, obj_id)
        if map_key in object_map:
            slot.set(object_map[map_key])
            slot.resolved = True
            continue
        groups.setdefault((kind, key), set()).add(obj_id)
        slot_targets.setdefault((kind, key, obj_id), []).append(slot)
    return groups, slot_targets


def prefetch_related(
    objects,
    max_depth=1,
    max_objects=None,
    max_queries_per_round=None,
    budget=None,
):
    """Eagerly dereference references reachable from ``objects`` under a budget.

    :param objects: a :class:`~mongoengine.document.Document`, an iterable of
        documents, or a :class:`~mongoengine.queryset.QuerySet`.
    :param max_depth: maximum number of reference hops to resolve (``None``
        for unlimited).
    :param max_objects: maximum number of unique referenced documents to pull
        from the database. The same document reached through several paths
        counts only once.
    :param max_queries_per_round: maximum number of batched queries issued in
        one round (one query per referenced collection). Remaining collections
        are retried in the next round without advancing the depth.
    :param budget: an existing :class:`PrefetchBudget`; when given it takes
        precedence over the individual keyword arguments.
    :returns: a :class:`PrefetchReport`. Documents are mutated in place;
        unresolved references are replaced with
        :class:`~mongoengine.base.datastructures.UnresolvedReference` markers.
    """
    if budget is None:
        budget = PrefetchBudget(
            max_depth=max_depth,
            max_objects=max_objects,
            max_queries_per_round=max_queries_per_round,
        )
    report = PrefetchReport(budget=budget)

    if isinstance(objects, QuerySet):
        objects = list(objects)
    elif isinstance(objects, Document):
        objects = [objects]
    else:
        objects = list(objects)

    # Every unique materialised document, keyed by (collection_name, pk).
    object_map = {}
    for root in objects:
        if isinstance(root, Document) and root.pk is not None:
            object_map.setdefault(_doc_identity(root), root)

    # All discovered slots that have not been finalised yet, keyed by path.
    all_slots = {}

    def discover(docs, doc_paths):
        """Scan the current frontier docs, register new slots and attach the
        ones that point to already-materialised documents (shared refs)."""
        scanned = []
        _collect_slots(docs, scanned, doc_paths)
        groups, slot_targets = _group_slots(scanned, object_map)
        for slot in scanned:
            if slot.resolved:
                report.resolved_paths.append(slot.path)
                all_slots.pop(slot.path, None)
            else:
                if slot.path not in all_slots:
                    all_slots[slot.path] = slot
                else:
                    # Same graph path rediscovered (should be rare); keep the
                    # first occurrence and mark the duplicate resolved.
                    slot.resolved = True
                    report.resolved_paths.append(slot.path)
        return groups, slot_targets

    def refresh_groups(groups, slot_targets):
        """Drop targets that have been fetched meanwhile (shared refs)."""
        fresh_groups = {}
        fresh_targets = {}
        for (kind, key), ids in groups.items():
            col_name = key._get_collection_name() if kind != "collection" else key
            for oid in ids:
                if (col_name, oid) in object_map:
                    doc = object_map[(col_name, oid)]
                    for slot in slot_targets.get((kind, key, oid), ()):
                        slot.set(doc)
                        slot.resolved = True
                        report.resolved_paths.append(slot.path)
                        all_slots.pop(slot.path, None)
                        carry_reasons.pop(slot.path, None)
                else:
                    fresh_groups.setdefault((kind, key), set()).add(oid)
                    fresh_targets.setdefault((kind, key, oid), []).extend(
                        slot_targets.get((kind, key, oid), ())
                    )
        return fresh_groups, fresh_targets

    frontier_docs = list(objects)
    frontier_paths = {
        id(doc): f"root[{index}]" for index, doc in enumerate(objects)
    }
    depth = 0
    # Work carried over because a per-round budget blocked it. These targets
    # belong to the current depth and must be drained before advancing.
    carry_groups = {}
    carry_targets = {}
    # Tentative unresolved reason for carried slots (queries/objects); only
    # slots surviving to finalisation are reported.
    carry_reasons = {}

    while True:
        if carry_groups:
            groups, slot_targets = refresh_groups(carry_groups, carry_targets)
            carry_groups, carry_targets = {}, {}
        else:
            if not frontier_docs:
                break
            depth += 1
            groups, slot_targets = discover(frontier_docs, frontier_paths)
            frontier_docs = []
            frontier_paths = {}
            # Discover one level past the depth limit purely so the boundary
            # references get marked (rather than left as raw DBRefs); do not
            # fetch anything beyond the limit.
            if budget.max_depth is not None and depth > budget.max_depth:
                break

        if not groups:
            continue

        target_keys = sorted(groups, key=lambda t: (t[0], str(t[1])))
        queries_this_round = 0
        fetched_docs = []
        fetched_paths = {}
        objects_exhausted = False

        for kind, key in target_keys:
            ids = groups[(kind, key)]
            col_name = key._get_collection_name() if kind != "collection" else key
            # Another group (e.g. a typed vs generic edge to the same
            # document) may have materialised some of these targets already.
            pending_ids = set()
            for oid in ids:
                if (col_name, oid) in object_map:
                    doc = object_map[(col_name, oid)]
                    for slot in slot_targets[(kind, key, oid)]:
                        slot.set(doc)
                        report.resolved_paths.append(slot.path)
                        all_slots.pop(slot.path, None)
                else:
                    pending_ids.add(oid)
            ids = pending_ids
            if not ids:
                continue

            target_slots = {
                oid: slot_targets[(kind, key, oid)] for oid in ids
            }

            if (
                budget.max_queries_per_round is not None
                and queries_this_round >= budget.max_queries_per_round
            ):
                carry_groups[(kind, key)] = set(ids)
                carry_targets.update(
                    {
                        (kind, key, oid): target_slots[oid]
                        for oid in ids
                    }
                )
                for oid in ids:
                    for slot in target_slots[oid]:
                        carry_reasons[slot.path] = REASON_QUERIES
                continue

            remaining_capacity = None
            if budget.max_objects is not None:
                remaining_capacity = budget.max_objects - report.objects_fetched
                if remaining_capacity <= 0:
                    objects_exhausted = True
                    for oid in ids:
                        for slot in target_slots[oid]:
                            carry_reasons[slot.path] = REASON_OBJECTS
                    break

            ordered_ids = sorted(ids, key=lambda value: str(value))
            if (
                remaining_capacity is not None
                and len(ordered_ids) > remaining_capacity
            ):
                query_ids = ordered_ids[:remaining_capacity]
                deferred_ids = ordered_ids[remaining_capacity:]
                objects_exhausted = True
            else:
                query_ids = ordered_ids
                deferred_ids = []

            try:
                if kind == "collection":
                    found = _fetch_raw_group(key, query_ids)
                else:
                    found = _fetch_group(key, query_ids)
            except Exception:
                for oid in query_ids:
                    for slot in target_slots[oid]:
                        report.unresolved[slot.path] = REASON_CLASS
                        all_slots.pop(slot.path, None)
                continue

            queries_this_round += 1
            report.query_count += 1

            for oid in query_ids:
                if oid in found:
                    doc = found[oid]
                    map_key = (col_name, oid)
                    if map_key not in object_map:
                        object_map[map_key] = doc
                        report.objects_fetched += 1
                        fetched_docs.append(doc)
                        # The first slot that reaches this document gives it
                        # the path prefix used when scanning the next depth.
                        fetched_paths.setdefault(id(doc), target_slots[oid][0].path)
                    else:
                        doc = object_map[map_key]
                    for slot in target_slots[oid]:
                        slot.set(doc)
                        report.resolved_paths.append(slot.path)
                        all_slots.pop(slot.path, None)
                else:
                    for slot in target_slots[oid]:
                        report.unresolved[slot.path] = REASON_MISSING
                        all_slots.pop(slot.path, None)

            if deferred_ids:
                carry_groups[(kind, key)] = set(deferred_ids)
                carry_targets.update(
                    {
                        (kind, key, oid): target_slots[oid]
                        for oid in deferred_ids
                    }
                )
                for oid in deferred_ids:
                    for slot in target_slots[oid]:
                        carry_reasons[slot.path] = REASON_OBJECTS

        if queries_this_round or not objects_exhausted:
            report.queries_by_round.append(queries_this_round)
            report.rounds += 1

        if carry_groups:
            # Slots carried on may include references to documents that were
            # fetched meanwhile (e.g. an edge closing a cycle back to a root).
            # Attach those now; only genuinely unresolved work is retried.
            carry_groups, carry_targets = refresh_groups(
                carry_groups, carry_targets
            )
            # Retry the blocked collections at the same depth.
            if carry_groups:
                continue
        if objects_exhausted:
            break
        frontier_docs = fetched_docs
        frontier_paths = fetched_paths

    _finalize_unresolved(all_slots, report, carry_reasons)
    _last_report_store.report = report
    return report


def _marker_for(slot, reason):
    kind, key, doc_cls, obj_id = slot.edge
    collection = (
        key._get_collection_name() if doc_cls is not None else key
    )
    return UnresolvedReference(
        document_type=doc_cls,
        pk=obj_id,
        collection=collection,
        path=slot.path,
        reason=reason,
    )


def _finalize_unresolved(all_slots, report, carry_reasons):
    """Replace every still-pending raw reference with an identifiable marker."""
    for path in sorted(all_slots):
        slot = all_slots[path]
        reason = report.unresolved.get(path) or carry_reasons.get(path)
        if reason is None:
            budget = report.budget
            # Slots still present here were discovered on the depth-boundary
            # scan and never fetched.
            reason = REASON_DEPTH if budget.max_depth is not None else REASON_QUERIES
        marker = _marker_for(slot, reason)
        slot.set(marker)
        report.unresolved[path] = reason
