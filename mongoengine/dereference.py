from bson import DBRef

from mongoengine.base import (
    BaseDict,
    BaseList,
    EmbeddedDocumentList,
    TopLevelDocumentMetaclass,
    _DocumentRegistry,
)
from mongoengine.base.datastructures import (
    BudgetedLazyReference,
    GenericBudgetedLazyReference,
    LazyReference,
)
from mongoengine.connection import _get_session, get_db
from mongoengine.document import Document, EmbeddedDocument
from mongoengine.fields import (
    DictField,
    ListField,
    MapField,
    ReferenceField,
)
from mongoengine.prefetch import PrefetchBudget, PrefetchReport, UnresolvedReference
from mongoengine.queryset import QuerySet


class DeReference:
    def __call__(
        self,
        items,
        max_depth=1,
        instance=None,
        name=None,
        budget=None,
        root_prefix=None,
    ):
        """
        Cheaply dereferences the items to a set depth.
        Also handles the conversion of complex data types.

        :param items: The iterable (dict, list, queryset) to be dereferenced.
        :param max_depth: The maximum depth to recurse to
        :param instance: The owning instance used for tracking changes by
            :class:`~mongoengine.base.ComplexBaseField`
        :param name: The name of the field, used for tracking changes by
            :class:`~mongoengine.base.ComplexBaseField`
        :param get: A boolean determining if being called by __get__
        """
        if items is None or isinstance(items, str):
            return items

        # cheapest way to convert a queryset to a list
        # list(queryset) uses a count() query to determine length
        if isinstance(items, QuerySet):
            items = [i for i in items]

        if budget is not None:
            if root_prefix is None:
                root_prefix = "[" if isinstance(items, list) else ""
            return self._budgeted_dereference(
                items, budget, root_prefix=root_prefix
            )

        self.max_depth = max_depth
        doc_type = None

        if instance and isinstance(
            instance, (Document, EmbeddedDocument, TopLevelDocumentMetaclass)
        ):
            doc_type = instance._fields.get(name)
            while hasattr(doc_type, "field"):
                doc_type = doc_type.field

            if isinstance(doc_type, ReferenceField):
                field = doc_type
                doc_type = doc_type.document_type
                is_list = not hasattr(items, "items")

                if is_list and all(i.__class__ == doc_type for i in items):
                    return items
                elif not is_list and all(
                    i.__class__ == doc_type for i in items.values()
                ):
                    return items
                elif not field.dbref:
                    # We must turn the ObjectIds into DBRefs

                    # Recursively dig into the sub items of a list/dict
                    # to turn the ObjectIds into DBRefs
                    def _get_items_from_list(items):
                        new_items = []
                        for v in items:
                            value = v
                            if isinstance(v, dict):
                                value = _get_items_from_dict(v)
                            elif isinstance(v, list):
                                value = _get_items_from_list(v)
                            elif not isinstance(v, (DBRef, Document)):
                                value = field.to_python(v)
                            new_items.append(value)
                        return new_items

                    def _get_items_from_dict(items):
                        new_items = {}
                        for k, v in items.items():
                            value = v
                            if isinstance(v, list):
                                value = _get_items_from_list(v)
                            elif isinstance(v, dict):
                                value = _get_items_from_dict(v)
                            elif not isinstance(v, (DBRef, Document)):
                                value = field.to_python(v)
                            new_items[k] = value
                        return new_items

                    if not hasattr(items, "items"):
                        items = _get_items_from_list(items)
                    else:
                        items = _get_items_from_dict(items)

        self.reference_map = self._find_references(items)
        self.object_map = self._fetch_objects(doc_type=doc_type)
        return self._attach_objects(items, 0, instance, name)

    def _budgeted_dereference(self, items, budget, root_prefix="["):
        roots = list(items)
        object_map = {}
        expanded = set()
        reasons = {}
        queries_executed = 0
        unique_roots = set()

        for index, root in enumerate(roots):
            if isinstance(root, Document) and root.pk is not None:
                root_key = (root._get_collection_name(), root.pk)
                object_map[root_key] = root
                unique_roots.add(root_key)

        frontier = [(root, 0) for root in roots]
        depths_reached = 0
        truncated_by = None
        object_slots_remaining = budget.max_unique_objects

        while frontier:
            current_depth = min(item_depth for _, item_depth in frontier)
            pending = {}
            group_index = {}
            next_frontier = []
            level_items = [
                (doc, item_depth)
                for doc, item_depth in frontier
                if item_depth == current_depth
            ]
            deferred_items = [
                (doc, item_depth)
                for doc, item_depth in frontier
                if item_depth != current_depth
            ]

            for doc, _ in level_items:
                if not isinstance(doc, Document) or doc.pk is None:
                    continue

                doc_key = (doc._get_collection_name(), doc.pk)
                if doc_key in expanded:
                    continue
                expanded.add(doc_key)

                for edge in self._budget_edges(
                    doc,
                    "",
                    current_depth + 1,
                    set(),
                    budget.max_depth,
                ):
                    target_key = (edge["collection"], edge["id"])
                    group_key = (
                        edge["depth"],
                        edge["collection"],
                        getattr(edge["document_type"], "__name__", None),
                    )
                    if group_key not in group_index:
                        group_index[group_key] = {
                            "depth": edge["depth"],
                            "collection": edge["collection"],
                            "document_type": edge["document_type"],
                            "ids": set(),
                        }
                        pending[group_key] = group_index[group_key]
                    group = group_index[group_key]
                    if target_key not in object_map:
                        group["ids"].add(edge["id"])

            groups_by_depth = {}
            for group in pending.values():
                groups_by_depth.setdefault(group["depth"], []).append(group)

            edge_depth = current_depth + 1
            groups = sorted(
                groups_by_depth.get(edge_depth, []),
                key=lambda group: (
                    group["collection"],
                    getattr(group["document_type"], "__name__", ""),
                ),
            )

            if budget.max_depth is not None and edge_depth > budget.max_depth:
                truncated_by = "max_depth"
                for group in groups:
                    for object_id in group["ids"]:
                        reasons.setdefault(
                            (group["collection"], object_id), "max_depth"
                        )
                frontier = deferred_items
                continue

            depth_queries = 0
            for group in groups:
                if (
                    budget.max_queries_per_round is not None
                    and depth_queries >= budget.max_queries_per_round
                ):
                    truncated_by = "max_queries_per_round"
                    break

                ids = sorted(group["ids"], key=lambda value: str(value))
                if object_slots_remaining is not None:
                    ids = ids[:object_slots_remaining]
                if not ids:
                    truncated_by = "max_unique_objects"
                    break

                if group["document_type"] is None:
                    group["document_type"] = self._document_type_for_collection(
                        group["collection"]
                    )
                if group["document_type"] is None:
                    for object_id in group["ids"]:
                        reasons.setdefault(
                            (group["collection"], object_id),
                            "unknown_document_type",
                        )
                    continue

                documents = self._fetch_budget_group(group, ids)
                depth_queries += 1
                queries_executed += 1
                if object_slots_remaining is not None:
                    object_slots_remaining -= len(ids)

                for object_id in ids:
                    target_key = (group["collection"], object_id)
                    document = documents.get(object_id)
                    if document is None:
                        reasons[target_key] = "not_found"
                    else:
                        object_map[target_key] = document
                        next_frontier.append((document, edge_depth))

                found_count = sum(
                    1
                    for object_id in documents
                    if (group["collection"], object_id) in object_map
                )
                if object_slots_remaining is not None:
                    object_slots_remaining += len(ids) - found_count
                if object_slots_remaining is not None and object_slots_remaining <= 0:
                    truncated_by = "max_unique_objects"
                    break

            if depth_queries or any(
                target_key in object_map
                for group in groups
                for target_key in [
                    (group["collection"], object_id) for object_id in group["ids"]
                ]
            ):
                depths_reached = max(depths_reached, edge_depth)

            if truncated_by:
                for group in groups:
                    for object_id in group["ids"]:
                        target_key = (group["collection"], object_id)
                        if target_key not in object_map:
                            reasons.setdefault(target_key, truncated_by)
                break
            frontier = deferred_items + next_frontier

        unresolved = []
        for index, root in enumerate(roots):
            prefix = f"{root_prefix}{index}]" if root_prefix else ""
            self._attach_budgeted(
                root,
                object_map,
                reasons,
                budget,
                prefix,
                0,
                set(),
                unresolved,
            )

        unique_objects_fetched = len(object_map) - len(
            unique_roots
        )
        report = PrefetchReport(
            max_depth=budget.max_depth,
            max_unique_objects=budget.max_unique_objects,
            max_queries_per_round=budget.max_queries_per_round,
            queries_executed=queries_executed,
            unique_objects_fetched=unique_objects_fetched,
            depths_reached=depths_reached,
            truncated_by=truncated_by,
            unresolved=tuple(unresolved),
        )
        return roots, report

    def _attach_budgeted(
        self,
        value,
        object_map,
        reasons,
        budget,
        path,
        depth,
        ancestors,
        unresolved,
        field=None,
    ):
        if value is None:
            return None

        if isinstance(value, LazyReference):
            return value

        if isinstance(value, (Document, EmbeddedDocument)):
            key = None
            if isinstance(value, Document) and value.pk is not None:
                key = (value._get_collection_name(), value.pk)
                if key in ancestors:
                    return value
                ancestors = ancestors | {key}

            for field_name in value._fields:
                child_field = value._fields[field_name]
                child_path = f"{path}.{field_name}" if path else field_name
                value._data[field_name] = self._attach_budgeted(
                    value._data.get(field_name),
                    object_map,
                    reasons,
                    budget,
                    child_path,
                    depth,
                    ancestors,
                    unresolved,
                    child_field,
                )
            return value

        if isinstance(value, DBRef):
            target_key = (value.collection, value.id)
            document = object_map.get(target_key)
            if document is not None:
                return self._attach_budgeted(
                    document,
                    object_map,
                    reasons,
                    budget,
                    path,
                    depth + 1,
                    ancestors,
                    unresolved,
                )

            reason = reasons.get(target_key)
            if reason is not None:
                if reason in (
                    "max_unique_objects",
                    "max_queries_per_round",
                    "max_depth",
                    "unknown_document_type",
                ):
                    document_type = self._edge_document_type(value, field)
                    if document_type is not None:
                        placeholder = BudgetedLazyReference(
                            document_type,
                            value.id,
                            path=path,
                            reason=reason,
                        )
                    else:
                        placeholder = GenericBudgetedLazyReference(
                            value.collection,
                            value.id,
                            path=path,
                            reason=reason,
                        )
                    unresolved.append(
                        UnresolvedReference(
                            path=path,
                            collection=value.collection,
                            document_id=value.id,
                            document_class=getattr(document_type, "__name__", None),
                            depth=depth + 1,
                            reason=reason,
                        )
                    )
                    return placeholder

                unresolved.append(
                    UnresolvedReference(
                        path=path,
                        collection=value.collection,
                        document_id=value.id,
                        document_class=getattr(
                            self._edge_document_type(value, field), "__name__", None
                        ),
                        depth=depth + 1,
                        reason=reason,
                    )
                )
            return value

        if isinstance(value, dict) and "_ref" in value:
            target_key = (value["_ref"].collection, value["_ref"].id)
            document = object_map.get(target_key)
            if document is not None:
                return self._attach_budgeted(
                    document,
                    object_map,
                    reasons,
                    budget,
                    path,
                    depth + 1,
                    ancestors,
                    unresolved,
                )

            reason = reasons.get(target_key)
            if reason in (
                "max_unique_objects",
                "max_queries_per_round",
                "max_depth",
                "unknown_document_type",
            ):
                try:
                    document_type = _DocumentRegistry.get(value["_cls"])
                    placeholder = BudgetedLazyReference(
                        document_type,
                        value["_ref"].id,
                        path=path,
                        reason=reason,
                    )
                    document_class = document_type.__name__
                except Exception:
                    placeholder = GenericBudgetedLazyReference(
                        value["_ref"].collection,
                        value["_ref"].id,
                        path=path,
                        reason=reason,
                        document_class_name=value["_cls"],
                    )
                    document_class = value["_cls"]
                unresolved.append(
                    UnresolvedReference(
                        path=path,
                        collection=value["_ref"].collection,
                        document_id=value["_ref"].id,
                        document_class=document_class,
                        depth=depth + 1,
                        reason=reason,
                    )
                )
                return placeholder
            if reason is not None:
                unresolved.append(
                    UnresolvedReference(
                        path=path,
                        collection=value["_ref"].collection,
                        document_id=value["_ref"].id,
                        document_class=value["_cls"],
                        depth=depth + 1,
                        reason=reason,
                    )
                )
            return value

        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                result[key] = self._attach_budgeted(
                    item,
                    object_map,
                    reasons,
                    budget,
                    child_path,
                    depth,
                    ancestors,
                    unresolved,
                    getattr(field, "field", field),
                )
            if isinstance(value, BaseDict):
                value.clear()
                value.update(result)
                value._dereferenced = True
                value._budgeted_dereferenced = True
                return value
            result = BaseDict(result, None, path or "$")
            result._dereferenced = True
            result._budgeted_dereferenced = True
            return result

        if isinstance(value, (list, tuple)):
            item_field = getattr(field, "field", field)
            result = []
            for index, item in enumerate(value):
                result.append(
                    self._attach_budgeted(
                        item,
                        object_map,
                        reasons,
                        budget,
                        f"{path}[{index}]",
                        depth,
                        ancestors,
                        unresolved,
                        item_field,
                    )
                )
            if isinstance(value, EmbeddedDocumentList):
                attached = EmbeddedDocumentList(result, value._instance, value._name)
            elif isinstance(value, BaseList):
                attached = BaseList(result, value._instance, value._name)
            else:
                attached = BaseList(result, None, path or "$")
            attached._dereferenced = True
            attached._budgeted_dereferenced = True
            if isinstance(value, tuple):
                return tuple(attached)
            return attached

        if isinstance(field, ReferenceField):
            document_type = field.document_type
            target_key = (document_type._get_collection_name(), value)
            document = object_map.get(target_key)
            if document is not None:
                return self._attach_budgeted(
                    document,
                    object_map,
                    reasons,
                    budget,
                    path,
                    depth + 1,
                    ancestors,
                    unresolved,
                )
            reason = reasons.get(target_key)
            if reason in ("max_unique_objects", "max_queries_per_round", "max_depth"):
                unresolved.append(
                    UnresolvedReference(
                        path=path,
                        collection=target_key[0],
                        document_id=value,
                        document_class=document_type.__name__,
                        depth=depth + 1,
                        reason=reason,
                    )
                )
                return BudgetedLazyReference(
                    document_type,
                    value,
                    path=path,
                    reason=reason,
                )
            if reason is not None:
                unresolved.append(
                    UnresolvedReference(
                        path=path,
                        collection=target_key[0],
                        document_id=value,
                        document_class=document_type.__name__,
                        depth=depth + 1,
                        reason=reason,
                    )
                )

        return value

    def _fetch_budget_group(self, group, ids):
        document_type = group["document_type"]
        collection_name = group["collection"]
        if document_type is None:
            document_type = self._document_type_for_collection(collection_name)
        if document_type is None:
            return {}
        return document_type.objects().in_bulk(ids)

    @staticmethod
    def _document_type_for_collection(collection_name):
        try:
            class_name = "".join(
                part.capitalize() for part in collection_name.split("_")
            )
            document_type = _DocumentRegistry.get(class_name)
        except Exception:
            return None
        if document_type._get_collection_name() == collection_name:
            return document_type
        return None

    def _budget_edges(
        self, value, path, depth, scanned_documents=None, max_depth=None
    ):
        if scanned_documents is None:
            scanned_documents = set()

        if isinstance(value, (Document, EmbeddedDocument)):
            if (
                max_depth is None
                and isinstance(value, Document)
                and value.pk is not None
            ):
                document_key = (value._get_collection_name(), value.pk)
                if document_key in scanned_documents:
                    return []
                scanned_documents.add(document_key)

            edges = []
            for field_name, field in value._fields.items():
                field_value = value._data.get(field_name)
                field_path = f"{path}.{field_name}" if path else field_name
                if field_value is None:
                    continue
                edges.extend(
                    self._budget_edges_for_value(
                        field_value,
                        field_path,
                        depth,
                        field,
                        scanned_documents,
                        max_depth,
                    )
                )
            return edges
        return self._budget_edges_for_value(
            value, path, depth, None, scanned_documents, max_depth
        )

    def _budget_edges_for_value(
        self, value, path, depth, field, scanned_documents, max_depth=None
    ):
        if value is None:
            return []

        if isinstance(value, LazyReference):
            return []

        if isinstance(value, (Document, EmbeddedDocument)):
            return self._budget_edges(
                value, path, depth, scanned_documents, max_depth
            )

        if isinstance(value, DBRef):
            document_type = self._edge_document_type(value, field)
            return [
                {
                    "path": path,
                    "collection": value.collection,
                    "id": value.id,
                    "document_type": document_type,
                    "depth": depth,
                }
            ]

        if isinstance(value, dict) and "_ref" in value:
            document_type = _DocumentRegistry.get(value["_cls"])
            return [
                {
                    "path": path,
                    "collection": value["_ref"].collection,
                    "id": value["_ref"].id,
                    "document_type": document_type,
                    "depth": depth,
                }
            ]

        if isinstance(value, dict):
            edges = []
            for key, item in value.items():
                edges.extend(
                    self._budget_edges_for_value(
                        item,
                        f"{path}.{key}" if path else str(key),
                        depth,
                        field,
                        scanned_documents,
                        max_depth,
                    )
                )
            return edges

        if isinstance(value, (list, tuple)):
            edges = []
            item_field = getattr(field, "field", field)
            for index, item in enumerate(value):
                edges.extend(
                    self._budget_edges_for_value(
                        item,
                        f"{path}[{index}]",
                        depth,
                        item_field,
                        scanned_documents,
                        max_depth,
                    )
                )
            return edges

        if isinstance(field, ReferenceField):
            document_type = field.document_type
            return [
                {
                    "path": path,
                    "collection": document_type._get_collection_name(),
                    "id": value,
                    "document_type": document_type,
                    "depth": depth,
                }
            ]

        return []

    @staticmethod
    def _edge_document_type(dbref, field):
        class_name = getattr(dbref, "cls", None)
        if class_name:
            try:
                return _DocumentRegistry.get(class_name)
            except Exception:
                pass
        if isinstance(field, ReferenceField):
            return field.document_type
        return DeReference._document_type_for_collection(dbref.collection)

    def _find_references(self, items, depth=0):
        """
        Recursively finds all db references to be dereferenced

        :param items: The iterable (dict, list, queryset)
        :param depth: The current depth of recursion
        """
        reference_map = {}
        if not items or depth >= self.max_depth:
            return reference_map

        # Determine the iterator to use
        if isinstance(items, dict):
            iterator = items.values()
        else:
            iterator = items

        # Recursively find dbreferences
        depth += 1
        for item in iterator:
            if isinstance(item, (Document, EmbeddedDocument)):
                for field_name, field in item._fields.items():
                    v = item._data.get(field_name, None)
                    if isinstance(v, LazyReference):
                        # LazyReference inherits DBRef but should not be dereferenced here !
                        continue
                    elif isinstance(v, DBRef):
                        reference_map.setdefault(field.document_type, set()).add(v.id)
                    elif isinstance(v, dict) and "_ref" in v:
                        reference_map.setdefault(
                            _DocumentRegistry.get(v["_cls"]), set()
                        ).add(v["_ref"].id)
                    elif isinstance(v, (dict, list, tuple)) and depth <= self.max_depth:
                        field_cls = getattr(
                            getattr(field, "field", None), "document_type", None
                        )
                        references = self._find_references(v, depth)
                        for key, refs in references.items():
                            if isinstance(
                                field_cls, (Document, TopLevelDocumentMetaclass)
                            ):
                                key = field_cls
                            reference_map.setdefault(key, set()).update(refs)
            elif isinstance(item, LazyReference):
                # LazyReference inherits DBRef but should not be dereferenced here !
                continue
            elif isinstance(item, DBRef):
                reference_map.setdefault(item.collection, set()).add(item.id)
            elif isinstance(item, dict) and "_ref" in item:
                reference_map.setdefault(
                    _DocumentRegistry.get(item["_cls"]), set()
                ).add(item["_ref"].id)
            elif isinstance(item, (dict, list, tuple)) and depth - 1 <= self.max_depth:
                references = self._find_references(item, depth - 1)
                for key, refs in references.items():
                    reference_map.setdefault(key, set()).update(refs)

        return reference_map

    def _fetch_objects(self, doc_type=None):
        """Fetch all references and convert to their document objects"""
        object_map = {}
        for collection, dbrefs in self.reference_map.items():
            # we use getattr instead of hasattr because hasattr swallows any exception under python2
            # so it could hide nasty things without raising exceptions (cfr bug #1688))
            ref_document_cls_exists = getattr(collection, "objects", None) is not None

            if ref_document_cls_exists:
                col_name = collection._get_collection_name()
                refs = [
                    dbref for dbref in dbrefs if (col_name, dbref) not in object_map
                ]
                references = collection.objects().in_bulk(refs)
                for key, doc in references.items():
                    object_map[(col_name, key)] = doc
            else:  # Generic reference: use the refs data to convert to document
                if isinstance(doc_type, (ListField, DictField, MapField)):
                    continue

                refs = [
                    dbref for dbref in dbrefs if (collection, dbref) not in object_map
                ]

                if doc_type:
                    references = doc_type._get_db()[collection].find(
                        {"_id": {"$in": refs}}, session=_get_session()
                    )
                    for ref in references:
                        doc = doc_type._from_son(ref)
                        object_map[(collection, doc.id)] = doc
                else:
                    references = get_db()[collection].find(
                        {"_id": {"$in": refs}}, session=_get_session()
                    )
                    for ref in references:
                        if "_cls" in ref:
                            doc = _DocumentRegistry.get(ref["_cls"])._from_son(ref)
                        elif doc_type is None:
                            doc = _DocumentRegistry.get(
                                "".join(x.capitalize() for x in collection.split("_"))
                            )._from_son(ref)
                        else:
                            doc = doc_type._from_son(ref)
                        object_map[(collection, doc.id)] = doc
        return object_map

    def _attach_objects(self, items, depth=0, instance=None, name=None):
        """
        Recursively finds all db references to be dereferenced

        :param items: The iterable (dict, list, queryset)
        :param depth: The current depth of recursion
        :param instance: The owning instance used for tracking changes by
            :class:`~mongoengine.base.ComplexBaseField`
        :param name: The name of the field, used for tracking changes by
            :class:`~mongoengine.base.ComplexBaseField`
        """
        if not items:
            if isinstance(items, (BaseDict, BaseList)):
                return items

            if instance:
                if isinstance(items, dict):
                    return BaseDict(items, instance, name)
                else:
                    return BaseList(items, instance, name)

        if isinstance(items, dict):
            if "_ref" in items:
                return self.object_map.get(
                    (items["_ref"].collection, items["_ref"].id), items
                )
            elif "_cls" in items:
                doc = _DocumentRegistry.get(items["_cls"])._from_son(items)
                _cls = doc._data.pop("_cls", None)
                del items["_cls"]
                doc._data = self._attach_objects(doc._data, depth, doc, None)
                if _cls is not None:
                    doc._data["_cls"] = _cls
                return doc

        if not hasattr(items, "items"):
            is_list = True
            list_type = BaseList
            if isinstance(items, EmbeddedDocumentList):
                list_type = EmbeddedDocumentList
            as_tuple = isinstance(items, tuple)
            iterator = enumerate(items)
            data = []
        else:
            is_list = False
            iterator = items.items()
            data = {}

        depth += 1
        for k, v in iterator:
            if is_list:
                data.append(v)
            else:
                data[k] = v

            if k in self.object_map and not is_list:
                data[k] = self.object_map[k]
            elif isinstance(v, (Document, EmbeddedDocument)):
                for field_name in v._fields:
                    v = data[k]._data.get(field_name, None)
                    if isinstance(v, DBRef):
                        data[k]._data[field_name] = self.object_map.get(
                            (v.collection, v.id), v
                        )
                    elif isinstance(v, dict) and "_ref" in v:
                        data[k]._data[field_name] = self.object_map.get(
                            (v["_ref"].collection, v["_ref"].id), v
                        )
                    elif isinstance(v, (dict, list, tuple)) and depth <= self.max_depth:
                        item_name = f"{name}.{k}.{field_name}"
                        data[k]._data[field_name] = self._attach_objects(
                            v, depth, instance=instance, name=item_name
                        )
            elif isinstance(v, (dict, list, tuple)) and depth <= self.max_depth:
                item_name = f"{name}.{k}" if name else name
                data[k] = self._attach_objects(
                    v, depth - 1, instance=instance, name=item_name
                )
            elif isinstance(v, DBRef) and hasattr(v, "id"):
                data[k] = self.object_map.get((v.collection, v.id), v)

        if instance and name:
            if is_list:
                return tuple(data) if as_tuple else list_type(data, instance, name)
            return BaseDict(data, instance, name)
        depth += 1
        return data
