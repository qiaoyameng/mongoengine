"""Keyed (business-key based) update generation for
:class:`~mongoengine.fields.EmbeddedDocumentListField`.

When an embedded document list is declared with ``key=<field name>``, its
elements own a stable business key.  Saving the parent document then diffs
the current list against the last persisted snapshot and, when the change
can be expressed safely, emits updates that target elements by key with
``arrayFilters`` instead of relying on array positions.

Targeting elements by key makes an update robust against array
re-indexing (reordering or removal that shifts indices).  Changes that
cannot be expressed safely (elements inserted in the middle, surviving
elements reordered together with other operations, a key mutated in place,
...) are represented as a full array replacement guarded by a precondition
ensuring the persisted array still matches the snapshot, so concurrent
modifications are never silently overwritten.
"""

__all__ = (
    "KeyedListUpdate",
    "build_keyed_list_update",
    "keyed_snapshot",
)


class _FallBack(Exception):
    """Internal signal: the diff cannot safely be expressed with
    positional-free operators."""


class KeyedListUpdate:
    """Result of diffing a keyed embedded document list.

    When ``safe`` is ``True``, ``update`` is a MongoDB update document that
    only uses filtered positional operators (and ``array_filters`` the
    matching arrayFilters).

    When ``safe`` is ``False``, ``replacement`` is the serialized array to
    write and ``condition`` is an optional filter document to AND into the
    save query as a precondition.

    ``changed`` is ``False`` when the current list is identical to the
    snapshot.
    """

    def __init__(
        self,
        *,
        db_field,
        safe,
        update=None,
        array_filters=None,
        replacement=None,
        condition=None,
        changed=True,
    ):
        self.db_field = db_field
        self.safe = safe
        self.update = update or {}
        self.array_filters = array_filters or []
        self.replacement = replacement
        self.condition = condition
        self.changed = changed


def keyed_snapshot(value, field):
    """Serialize a keyed list to its last-persisted representation."""
    return field.to_mongo(list(value))


def _duplicate_keys(items, key_db_field):
    seen = set()
    duplicates = set()
    for item in items:
        key_value = item[key_db_field]
        if key_value in seen:
            duplicates.add(key_value)
        seen.add(key_value)
    return duplicates


def _element_delta(doc):
    """Return ``(set_paths, unset_paths)`` for one surviving element."""
    set_data, unset_data = doc._delta()
    return set_data, list(unset_data)


def _has_mid_insertion(current_mongo, baseline_keys, key_db_field):
    """True if new (non-baseline) elements appear before surviving ones."""
    seen_new = False
    for item in current_mongo:
        is_new = item[key_db_field] not in baseline_keys
        if is_new:
            seen_new = True
        elif seen_new:
            return True
    return False


def _fallback(db_field, current_mongo, snapshot):
    """Build a guarded full-array replacement plan.

    The replacement is guarded by an equality match on the whole persisted
    array: any concurrent mutation (reorder, removal, insertion or field
    change) makes the filter mismatch and the save is rejected instead of
    overwriting the concurrent change.
    """
    return KeyedListUpdate(
        db_field=db_field,
        safe=False,
        replacement=current_mongo,
        condition={db_field: snapshot},
    )


def _identifier(db_field, position):
    # Filtered positional identifiers must start with a lowercase letter
    # and contain only lowercase alphanumeric characters.  Including the
    # field name keeps identifiers unique when several keyed lists are
    # updated in the same operation.
    sanitized = "".join(ch for ch in db_field.lower() if ch.isalnum())
    return f"el{sanitized}{position}"


def build_keyed_list_update(field, db_field, key_name, current, snapshot, snapshot_id=None):
    """Diff ``current`` against the persisted ``snapshot``.

    :param field: the :class:`EmbeddedDocumentListField`
    :param db_field: database name of the parent field
    :param key_name: python name of the business key field
    :param current: the live :class:`EmbeddedDocumentList`
    :param snapshot: last persisted serialized array (or ``None``)
    """
    from mongoengine.errors import FieldDoesNotExist, InvalidDocumentError, ValidationError

    document_type = field.field.document_type
    if key_name not in document_type._fields:
        raise FieldDoesNotExist(
            "Field '%s' is not a field of '%s' and cannot be used as the "
            "business key of '%s'"
            % (key_name, document_type._class_name, field.name)
        )
    key_db_field = document_type._db_field_map.get(key_name, key_name)

    current_mongo = [field.field.to_mongo(doc) for doc in current]

    duplicates = _duplicate_keys(current_mongo, key_db_field)
    if duplicates:
        raise ValidationError(
            "The business key '%s' of field '%s' must be unique; duplicated "
            "key value(s): %s"
            % (key_name, field.name, ", ".join(sorted(map(repr, duplicates))))
        )

    if snapshot is None:
        # Whole-list assignment (or no known baseline): nothing to diff.
        return KeyedListUpdate(
            db_field=db_field, safe=False, replacement=current_mongo
        )

    snapshot_duplicates = _duplicate_keys(snapshot, key_db_field)
    if snapshot_duplicates:
        raise InvalidDocumentError(
            "The persisted array of field '%s' contains duplicated business "
            "key value(s) for '%s': %s"
            % (
                field.name,
                key_name,
                ", ".join(sorted(map(repr, snapshot_duplicates))),
            )
        )

    baseline_by_key = {item[key_db_field]: item for item in snapshot}
    baseline_keys = set(baseline_by_key)
    baseline_order = [item[key_db_field] for item in snapshot]

    def key_of(doc):
        return field.field.to_mongo(doc)[key_db_field]

    current_keys = [key_of(doc) for doc in current]
    current_key_set = set(current_keys)
    removed_keys = [k for k in baseline_order if k not in current_key_set]
    new_keys = [k for k in dict.fromkeys(current_keys) if k not in baseline_keys]

    # The business key of an existing (persisted) element must never be
    # mutated in place.  Elements that were part of the snapshot are tracked
    # by object identity (``_keyed_baseline_id``); elements appended after
    # the snapshot are new and may still receive their key.
    for doc, key_value in zip(current, current_keys):
        if getattr(doc, "_keyed_baseline_id", None) != snapshot_id:
            continue
        if (
            key_db_field in doc._changed_fields
            or key_name in doc._changed_fields
            or key_db_field in doc._unset_fields
            or key_name in doc._unset_fields
        ):
            raise ValidationError(
                "The business key '%s' of an existing element of field "
                "'%s' cannot be changed in place" % (key_name, field.name)
            )

    survivor_order = [k for k in current_keys if k in baseline_keys]
    persisted_survivor_order = [k for k in baseline_order if k in current_key_set]
    survivors_reordered = survivor_order != persisted_survivor_order
    mid_insertion = _has_mid_insertion(current_mongo, baseline_keys, key_db_field)

    # A surviving key must still be carried by the exact same embedded
    # document object as in the snapshot.  Replacing an element in place
    # (``list[i] = new_doc`` with the same key) cannot be diffed field by
    # field, so it uses the guarded whole-array replacement.
    identity_replaced = False
    for doc, key_value in zip(current, current_keys):
        if key_value not in baseline_keys:
            continue
        if getattr(doc, "_keyed_baseline_id", None) != snapshot_id:
            identity_replaced = True
            break

    # Per-element field modifications, expressed with filtered positional
    # operators: they target elements by key and are therefore unaffected
    # by any shift of their position.
    set_ops = {}
    unset_ops = {}
    array_filters = []
    for position, doc in enumerate(current):
        key_value = key_of(doc)
        if key_value not in baseline_by_key:
            continue
        element_set, element_unset = _element_delta(doc)
        if not element_set and not element_unset:
            continue
        identifier = _identifier(db_field, position)
        for op_path, value in element_set.items():
            set_ops[f"{db_field}.$[{identifier}].{op_path}"] = value
        for op_path in element_unset:
            unset_ops[f"{db_field}.$[{identifier}].{op_path}"] = 1
        array_filters.append({f"{identifier}.{key_db_field}": key_value})

    field_modifications = bool(set_ops or unset_ops)
    appends = not survivors_reordered and not mid_insertion and bool(new_keys)

    try:
        if mid_insertion or identity_replaced:
            raise _FallBack()

        # A change of order among surviving elements cannot be combined
        # with any structural or field change (the in-memory final order
        # would not be reproducible).
        if survivors_reordered and (
            removed_keys or new_keys or field_modifications
        ):
            raise _FallBack()

        # Simultaneous removal and addition is ambiguous (the new elements
        # might replace removed positions); use the guarded replacement.
        if removed_keys and new_keys:
            raise _FallBack()

        update = {}

        if removed_keys:
            # $pull matches elements by key and never relies on positions.
            pull_query = (
                removed_keys[0]
                if len(removed_keys) == 1
                else {"$in": removed_keys}
            )
            update["$pull"] = {db_field: {key_db_field: pull_query}}

        if appends:
            # Only a tail append preserves every existing position.
            appended_keys = set(new_keys)
            appended = [
                item for item in current_mongo if item[key_db_field] in appended_keys
            ]
            update["$push"] = {db_field: {"$each": appended}}

        if set_ops:
            update["$set"] = set_ops
        if unset_ops:
            update["$unset"] = unset_ops

        if survivors_reordered:
            # Pure reordering: no positional-free operator can express it.
            return _fallback(db_field, current_mongo, snapshot)

        if not update:
            return KeyedListUpdate(db_field=db_field, safe=True, changed=False)

        return KeyedListUpdate(
            db_field=db_field,
            safe=True,
            update=update,
            array_filters=array_filters,
        )
    except _FallBack:
        return _fallback(db_field, current_mongo, snapshot)
