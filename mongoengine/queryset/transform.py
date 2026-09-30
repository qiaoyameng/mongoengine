import re
from collections import defaultdict

import pymongo
from bson import ObjectId
from bson.dbref import DBRef

from mongoengine.base import UPDATE_OPERATORS
from mongoengine.common import _import_class
from mongoengine.errors import InvalidQueryError

__all__ = ("query", "update", "array_filters", "STRING_OPERATORS")

# Matches a filtered positional operator placeholder, e.g. ``$[element]``
# in a Mongo update path.
ARRAY_FILTER_PLACEHOLDER_RE = re.compile(r"^\$\[(\w+)\]$")

# Matches the Django-style counterpart of a filtered positional operator,
# e.g. ``S[element]`` in an update keyword argument.
ARRAY_FILTER_SHORTCUT_RE = re.compile(r"^S\[(\w+)\]$")

COMPARISON_OPERATORS = (
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in",
    "nin",
    "mod",
    "all",
    "size",
    "exists",
    "not",
    "elemMatch",
    "type",
)
GEO_OPERATORS = (
    "within_distance",
    "within_spherical_distance",
    "within_box",
    "within_polygon",
    "near",
    "near_sphere",
    "max_distance",
    "min_distance",
    "geo_within",
    "geo_within_box",
    "geo_within_polygon",
    "geo_within_center",
    "geo_within_sphere",
    "geo_intersects",
)
STRING_OPERATORS = (
    "contains",
    "icontains",
    "startswith",
    "istartswith",
    "endswith",
    "iendswith",
    "exact",
    "iexact",
    "regex",
    "iregex",
    "wholeword",
    "iwholeword",
)
CUSTOM_OPERATORS = ("match",)
MATCH_OPERATORS = (
    COMPARISON_OPERATORS + GEO_OPERATORS + STRING_OPERATORS + CUSTOM_OPERATORS
)


def handle_raw_query(value, mongo_query):
    """Combine a raw query with an existing one"""
    for op, v in value.items():
        if op not in mongo_query:
            mongo_query[op] = v
        elif op in mongo_query and isinstance(mongo_query[op], dict):
            mongo_query[op].update(v)


# TODO make this less complex
def query(_doc_cls=None, **kwargs):
    """Transform a query from Django-style format to Mongo format."""
    mongo_query = {}
    merge_query = defaultdict(list)
    for key, value in sorted(kwargs.items()):
        if key == "__raw__":
            handle_raw_query(value, mongo_query)
            continue

        parts = key.rsplit("__")
        indices = [(i, p) for i, p in enumerate(parts) if p.isdigit()]
        parts = [part for part in parts if not part.isdigit()]
        # Check for an operator and transform to mongo-style if there is
        op = None
        if len(parts) > 1 and parts[-1] in MATCH_OPERATORS:
            op = parts.pop()

        # Allow to escape operator-like field name by __
        if len(parts) > 1 and parts[-1] == "":
            parts.pop()

        negate = False
        if len(parts) > 1 and parts[-1] == "not":
            parts.pop()
            negate = True

        if _doc_cls:
            # Switch field names to proper names [set in Field(name='foo')]
            try:
                fields = _doc_cls._lookup_field(parts)
            except Exception as e:
                raise InvalidQueryError(e)
            parts = []

            CachedReferenceField = _import_class("CachedReferenceField")
            GenericReferenceField = _import_class("GenericReferenceField")

            cleaned_fields = []
            for field in fields:
                append_field = True
                if isinstance(field, str):
                    parts.append(field)
                    append_field = False
                # is last and CachedReferenceField
                elif isinstance(field, CachedReferenceField) and fields[-1] == field:
                    parts.append("%s._id" % field.db_field)
                else:
                    parts.append(field.db_field)

                if append_field:
                    cleaned_fields.append(field)

            # Convert value to proper value
            field = cleaned_fields[-1]

            singular_ops = [None, "ne", "gt", "gte", "lt", "lte", "not"]
            singular_ops += STRING_OPERATORS
            is_iterable = False
            if op in singular_ops:
                value = field.prepare_query_value(op, value)

                if isinstance(field, CachedReferenceField) and value:
                    value = value["_id"]

            elif op in ("in", "nin", "all", "near") and not isinstance(value, dict):
                is_iterable = True
                # Raise an error if the in/nin/all/near param is not iterable.
                value = _prepare_query_for_iterable(field, op, value)

            # If we're querying a GenericReferenceField, we need to alter the
            # key depending on the value:
            # * If the value is a DBRef, the key should be "field_name._ref".
            # * If the value is an ObjectId, the key should be "field_name._ref.$id".
            if isinstance(field, GenericReferenceField):
                if isinstance(value, DBRef) or (
                    is_iterable and all(isinstance(v, DBRef) for v in value)
                ):
                    parts[-1] += "._ref"
                elif isinstance(value, ObjectId) or (
                    is_iterable and all(isinstance(v, ObjectId) for v in value)
                ):
                    parts[-1] += "._ref.$id"
                elif (
                    is_iterable
                    and any(isinstance(v, DBRef) for v in value)
                    and any(isinstance(v, ObjectId) for v in value)
                ):
                    raise ValueError(
                        "The `in`, `nin`, `all`, or `near`-operators cannot "
                        "be applied to mixed queries of DBRef/ObjectId/%s"
                        % _doc_cls.__name__
                    )

        # if op and op not in COMPARISON_OPERATORS:
        if op:
            if op in GEO_OPERATORS:
                value = _geo_operator(field, op, value)
            elif op in ("match", "elemMatch"):
                ListField = _import_class("ListField")
                EmbeddedDocumentField = _import_class("EmbeddedDocumentField")
                if (
                    isinstance(value, dict)
                    and isinstance(field, ListField)
                    and isinstance(field.field, EmbeddedDocumentField)
                ):
                    value = query(field.field.document_type, **value)
                else:
                    value = field.prepare_query_value(op, value)
                value = {"$elemMatch": value}
            elif op in CUSTOM_OPERATORS:
                NotImplementedError(
                    'Custom method "%s" has not ' "been implemented" % op
                )
            elif op not in STRING_OPERATORS:
                value = {"$" + op: value}

        if negate:
            value = {"$not": value}

        for i, part in indices:
            parts.insert(i, part)

        key = ".".join(parts)

        if key not in mongo_query:
            mongo_query[key] = value
        else:
            if isinstance(mongo_query[key], dict) and isinstance(value, dict):
                mongo_query[key].update(value)
                # $max/minDistance needs to come last - rebuild in order
                value_dict = mongo_query[key]
                if ("$maxDistance" in value_dict or "$minDistance" in value_dict) and (
                    "$near" in value_dict or "$nearSphere" in value_dict
                ):
                    ordered_value = {}
                    for k, v in value_dict.items():
                        if k == "$maxDistance" or k == "$minDistance":
                            continue
                        ordered_value[k] = v
                    # Required for MongoDB >= 2.6, may fail when combining
                    # PyMongo 3+ and MongoDB < 2.6
                    near_embedded = False
                    for near_op in ("$near", "$nearSphere"):
                        if isinstance(value_dict.get(near_op), dict):
                            ordered_value[near_op] = dict(ordered_value[near_op])
                            if "$maxDistance" in value_dict:
                                ordered_value[near_op]["$maxDistance"] = value_dict[
                                    "$maxDistance"
                                ]
                            if "$minDistance" in value_dict:
                                ordered_value[near_op]["$minDistance"] = value_dict[
                                    "$minDistance"
                                ]
                            near_embedded = True

                    if not near_embedded:
                        if "$maxDistance" in value_dict:
                            ordered_value["$maxDistance"] = value_dict["$maxDistance"]
                        if "$minDistance" in value_dict:
                            ordered_value["$minDistance"] = value_dict["$minDistance"]
                    mongo_query[key] = ordered_value
            else:
                # Store for manually merging later
                merge_query[key].append(value)

    # The queryset has been filter in such a way we must manually merge
    for k, v in merge_query.items():
        merge_query[k].append(mongo_query[k])
        del mongo_query[k]
        if isinstance(v, list):
            value = [{k: val} for val in v]
            if "$and" in mongo_query.keys():
                mongo_query["$and"].extend(value)
            else:
                mongo_query["$and"] = value

    return mongo_query


def update(_doc_cls=None, **update):
    """Transform an update spec from Django-style format to Mongo
    format.
    """
    mongo_update = {}

    for key, value in update.items():
        if key == "__raw__":
            handle_raw_query(value, mongo_update)
            continue

        parts = key.split("__")

        # if there is no operator, default to 'set'
        if len(parts) < 3 and parts[0] not in UPDATE_OPERATORS:
            parts.insert(0, "set")

        # Check for an operator and transform to mongo-style if there is
        op = None
        if parts[0] in UPDATE_OPERATORS:
            op = parts.pop(0)
            # Convert Pythonic names to Mongo equivalents
            operator_map = {
                "push_all": "pushAll",
                "pull_all": "pullAll",
                "dec": "inc",
                "add_to_set": "addToSet",
                "set_on_insert": "setOnInsert",
            }
            if op == "dec":
                # Support decrement by flipping a positive value's sign
                # and using 'inc'
                value = -value
            # If the operator doesn't found from operator map, the op value
            # will stay unchanged
            op = operator_map.get(op, op)

        match = None

        if len(parts) == 1:
            # typical update like set__field
            # but also allows to update a field named like a comparison operator
            # like set__type = "something" (without clashing with the 'type' operator)
            pass
        elif len(parts) > 1:
            # can be either an embedded field like set__foo__bar
            # or a comparison operator as in pull__foo__in
            if parts[-1] in COMPARISON_OPERATORS:
                match = parts.pop()  # e.g. pop 'in' from pull__foo__in

            # Allow to escape operator-like field name by __
            # e.g. in the case of an embedded foo.type field
            # Doc.objects().update(set__foo__type="bar")
            # see https://github.com/MongoEngine/mongoengine/pull/1351
            if parts[-1] == "":
                match = parts.pop()  # e.g. pop last '__' from set__foo__type__

        if _doc_cls:
            # Switch field names to proper names [set in Field(name='foo')]
            try:
                fields = _doc_cls._lookup_field(parts)
            except Exception as e:
                raise InvalidQueryError(e)
            parts = []

            cleaned_fields = []
            appended_sub_field = False
            for field in fields:
                append_field = True
                if isinstance(field, str):
                    # Convert the S operator to $
                    if field == "S":
                        field = "$"
                    else:
                        # Convert S[identifier] to a filtered positional
                        # operator, e.g. S[element] -> $[element]
                        array_filter_match = ARRAY_FILTER_SHORTCUT_RE.match(field)
                        if array_filter_match:
                            field = "$[%s]" % array_filter_match.group(1)
                    parts.append(field)
                    append_field = False
                else:
                    parts.append(field.db_field)
                if append_field:
                    appended_sub_field = False
                    cleaned_fields.append(field)
                    if hasattr(field, "field"):
                        cleaned_fields.append(field.field)
                        appended_sub_field = True

            # Convert value to proper value
            if appended_sub_field:
                field = cleaned_fields[-2]
            else:
                field = cleaned_fields[-1]

            GeoJsonBaseField = _import_class("GeoJsonBaseField")
            if isinstance(field, GeoJsonBaseField):
                value = field.to_mongo(value)

            if op == "pull":
                if field.required or value is not None:
                    if match in ("in", "nin") and not isinstance(value, dict):
                        value = _prepare_query_for_iterable(field, op, value)
                    else:
                        value = field.prepare_query_value(op, value)
            elif op == "push" and isinstance(value, (list, tuple, set)):
                value = [field.prepare_query_value(op, v) for v in value]
            elif op in (None, "set", "push"):
                if field.required or value is not None:
                    value = field.prepare_query_value(op, value)
            elif op in ("pushAll", "pullAll"):
                value = [field.prepare_query_value(op, v) for v in value]
            elif op in ("addToSet", "setOnInsert"):
                if isinstance(value, (list, tuple, set)):
                    value = [field.prepare_query_value(op, v) for v in value]
                elif field.required or value is not None:
                    value = field.prepare_query_value(op, value)
            elif op == "unset":
                value = 1
            # dec is normalized to inc above.
            elif op in ("inc", "mul"):
                value = field.prepare_query_value(op, value)

        if match:
            match = "$" + match
            value = {match: value}

        key = ".".join(parts)

        if "pull" in op and "." in key:
            # Dot operators don't work on pull operations
            # unless they point to a list field
            # Otherwise it uses nested dict syntax
            if op == "pullAll":
                raise InvalidQueryError(
                    "pullAll operations only support a single field depth"
                )

            # Look for the last list field and use dot notation until there
            field_classes = [c.__class__ for c in cleaned_fields]
            field_classes.reverse()
            ListField = _import_class("ListField")
            EmbeddedDocumentListField = _import_class("EmbeddedDocumentListField")
            if ListField in field_classes or EmbeddedDocumentListField in field_classes:
                # Join all fields via dot notation to the last ListField or EmbeddedDocumentListField
                # Then process as normal
                if ListField in field_classes:
                    _check_field = ListField
                else:
                    _check_field = EmbeddedDocumentListField

                last_listField = len(cleaned_fields) - field_classes.index(_check_field)
                key = ".".join(parts[:last_listField])
                parts = parts[last_listField:]
                parts.insert(0, key)

            parts.reverse()
            for key in parts:
                value = {key: value}
        elif op == "addToSet" and isinstance(value, list):
            value = {key: {"$each": value}}
        elif op in ("push", "pushAll"):
            if parts[-1].isdigit():
                key = ".".join(parts[0:-1])
                position = int(parts[-1])
                # $position expects an iterable. If pushing a single value,
                # wrap it in a list.
                if not isinstance(value, (set, tuple, list)):
                    value = [value]
                value = {key: {"$each": value, "$position": position}}
            else:
                if op == "pushAll":
                    op = "push"  # convert to non-deprecated keyword
                    if not isinstance(value, (set, tuple, list)):
                        value = [value]
                    value = {key: {"$each": value}}
                else:
                    value = {key: value}
        else:
            value = {key: value}
        key = "$" + op
        if key not in mongo_update:
            mongo_update[key] = value
        elif key in mongo_update and isinstance(mongo_update[key], dict):
            mongo_update[key].update(value)

    return mongo_update


def array_filters(_doc_cls, update, array_filters):
    """Transform model-aware ``array_filters`` into MongoDB syntax.

    Placeholder variables used in the update paths (e.g. ``$[element]``)
    are first bound to the element type of the corresponding
    :class:`~mongoengine.fields.ListField`. Each array filter is then
    resolved against the bound type so that field aliases (``db_field``)
    and values are converted, exactly like regular queries.

    :param _doc_cls: The document class being updated.
    :param update: The already transformed (Mongo-style) update document.
    :param array_filters: A list of Django-style array filters, or a
        ``{"__raw__": [...]}`` dict to bypass the transformation entirely.
    """
    if isinstance(array_filters, dict):
        # Explicit raw mode: pass the filters through untouched so they
        # are never converted twice.
        if "__raw__" in array_filters:
            return array_filters["__raw__"]
        raise InvalidQueryError(
            "array_filters must be a list of filters (or {'__raw__': [...]})"
        )

    bindings = _array_filter_bindings(_doc_cls, update)

    if array_filters is None:
        if bindings:
            raise InvalidQueryError(
                "The update uses array filter placeholder(s) %s but no "
                "array_filters were provided"
                % ", ".join("$[%s]" % name for name in sorted(bindings))
            )
        return None

    used = set()
    transformed = []
    for array_filter in array_filters:
        mongo_filter = {}
        for key, value in array_filter.items():
            parts = key.split("__")
            variable = parts[0]
            if variable not in bindings:
                raise InvalidQueryError(
                    'Array filter variable "%s" is not declared by any '
                    "placeholder in the update paths" % variable
                )
            used.add(variable)
            element_field = bindings[variable]
            mongo_filter.update(
                _transform_array_filter(variable, element_field, parts[1:], value)
            )
        transformed.append(mongo_filter)

    unused = set(bindings) - used
    if unused:
        raise InvalidQueryError(
            "Array filter placeholder(s) %s declared in the update paths are "
            "not used by any array_filters entry"
            % ", ".join("$[%s]" % name for name in sorted(unused))
        )

    return transformed


def _array_filter_bindings(_doc_cls, update):
    """Bind each ``$[variable]`` placeholder found in the update paths to
    the element field of the list it iterates over.
    """
    bindings = {}
    for path in _iter_update_paths(update):
        _bind_array_filter_variables(_doc_cls, path, bindings)
    return bindings


def _iter_update_paths(update):
    """Yield the dotted paths targeted by a transformed update document."""
    if isinstance(update, (list, tuple)):
        # Update with an aggregation pipeline
        for stage in update:
            yield from _iter_update_paths(stage)
    elif isinstance(update, dict):
        for operator, spec in update.items():
            if (
                isinstance(operator, str)
                and operator.startswith("$")
                and isinstance(spec, dict)
            ):
                for path in spec:
                    if isinstance(path, str):
                        yield path


def _bind_array_filter_variables(_doc_cls, path, bindings):
    """Walk a Mongo update path along the document fields and record the
    element field each ``$[variable]`` placeholder refers to.
    """
    doc_cls = _doc_cls
    field = None
    for part in path.split("."):
        match = ARRAY_FILTER_PLACEHOLDER_RE.match(part)
        if match or part == "$" or part.isdigit():
            # A positional segment applies to the current list field
            element_field = getattr(field, "field", None)
            if element_field is None:
                return
            if match:
                _bind_array_filter_variable(
                    bindings, match.group(1), element_field, path
                )
            doc_cls = getattr(element_field, "document_type", None)
            field = element_field
            continue

        # List fields are transparent in field paths, descend into the
        # element document type when there is one
        while field is not None and doc_cls is None and hasattr(field, "field"):
            field = field.field
            doc_cls = getattr(field, "document_type", None)
        if doc_cls is None:
            return

        field = _lookup_field_by_db_field(doc_cls, part)
        if field is None:
            # Unresolvable (e.g. dynamic field), stop binding this path
            return
        doc_cls = getattr(field, "document_type", None)


def _lookup_field_by_db_field(doc_cls, name):
    """Look up a field on a document class by python or database name."""
    field = doc_cls._fields.get(name)
    if field is not None:
        return field
    for candidate in doc_cls._fields.values():
        if candidate.db_field == name:
            return candidate
    return None


def _bind_array_filter_variable(bindings, name, element_field, path):
    """Record a variable binding, ensuring a variable is never bound to
    conflicting element types.
    """
    existing = bindings.get(name)
    if existing is not None and not _same_array_element_type(existing, element_field):
        raise InvalidQueryError(
            'Array filter variable "$[%s]" is bound to conflicting types '
            "(%s and %s) in the update paths"
            % (
                name,
                _array_element_type_name(existing),
                _array_element_type_name(element_field),
            )
        )
    bindings[name] = element_field


def _same_array_element_type(field_a, field_b):
    if field_a is field_b:
        return True
    if type(field_a) is not type(field_b):
        return False
    return getattr(field_a, "document_type", None) == getattr(
        field_b, "document_type", None
    )


def _array_element_type_name(field):
    document_type = getattr(field, "document_type", None)
    if document_type is not None:
        return document_type.__name__
    return type(field).__name__


def _transform_array_filter(variable, element_field, rest, value):
    """Transform a single Django-style array filter entry into a
    Mongo-style ``{variable.path: value}`` mapping.
    """
    document_type = getattr(element_field, "document_type", None)
    if document_type is not None and rest:
        if len(rest) == 1 and rest[0] in MATCH_OPERATORS:
            raise InvalidQueryError(
                'Cannot apply operator "%s" to array filter variable "%s" '
                "without a field path" % (rest[0], variable)
            )
        # Resolve the remaining path and operators against the embedded
        # document type, then prefix the variable back
        sub_query = query(document_type, **{"__".join(rest): value})
        return {
            "%s.%s" % (variable, key): sub_value
            for key, sub_value in sub_query.items()
        }

    if rest:
        # An operator applied to the array element itself (e.g. tag__in)
        if len(rest) > 1 or rest[0] not in MATCH_OPERATORS:
            raise InvalidQueryError(
                'Cannot resolve "%s" on array filter variable "%s"'
                % ("__".join(rest), variable)
            )
        return {variable: _prepare_array_element_value(element_field, rest[0], value)}

    # Filter on the whole array element
    if isinstance(value, dict) and all(key.startswith("$") for key in value):
        value = {
            key: element_field.prepare_query_value(None, sub_value)
            for key, sub_value in value.items()
        }
    else:
        value = element_field.prepare_query_value(None, value)
    return {variable: value}


def _prepare_array_element_value(field, op, value):
    """Prepare the value of an operator applied to an array element,
    mirroring the conversion performed by :func:`query`.
    """
    singular_ops = [None, "ne", "gt", "gte", "lt", "lte", "not"]
    singular_ops += STRING_OPERATORS
    if op in singular_ops:
        value = field.prepare_query_value(op, value)
    elif op in ("in", "nin", "all", "near") and not isinstance(value, dict):
        value = _prepare_query_for_iterable(field, op, value)

    if op in ("match", "elemMatch"):
        value = {"$elemMatch": field.prepare_query_value(op, value)}
    elif op not in STRING_OPERATORS:
        value = {"$" + op: value}
    return value


def _geo_operator(field, op, value):
    """Helper to return the query for a given geo query."""
    if op == "max_distance":
        value = {"$maxDistance": value}
    elif op == "min_distance":
        value = {"$minDistance": value}
    elif field._geo_index == pymongo.GEO2D:
        if op == "within_distance":
            value = {"$within": {"$center": value}}
        elif op == "within_spherical_distance":
            value = {"$within": {"$centerSphere": value}}
        elif op == "within_polygon":
            value = {"$within": {"$polygon": value}}
        elif op == "near":
            value = {"$near": value}
        elif op == "near_sphere":
            value = {"$nearSphere": value}
        elif op == "within_box":
            value = {"$within": {"$box": value}}
        else:
            raise NotImplementedError(
                'Geo method "%s" has not been ' "implemented for a GeoPointField" % op
            )
    else:
        if op == "geo_within":
            value = {"$geoWithin": _infer_geometry(value)}
        elif op == "geo_within_box":
            value = {"$geoWithin": {"$box": value}}
        elif op == "geo_within_polygon":
            value = {"$geoWithin": {"$polygon": value}}
        elif op == "geo_within_center":
            value = {"$geoWithin": {"$center": value}}
        elif op == "geo_within_sphere":
            value = {"$geoWithin": {"$centerSphere": value}}
        elif op == "geo_intersects":
            value = {"$geoIntersects": _infer_geometry(value)}
        elif op == "near":
            value = {"$near": _infer_geometry(value)}
        else:
            raise NotImplementedError(
                'Geo method "{}" has not been implemented for a {} '.format(
                    op, field._name
                )
            )
    return value


def _infer_geometry(value):
    """Helper method that tries to infer the $geometry shape for a
    given value.
    """
    if isinstance(value, dict):
        if "$geometry" in value:
            return value
        elif "coordinates" in value and "type" in value:
            return {"$geometry": value}
        raise InvalidQueryError(
            "Invalid $geometry dictionary should have type and coordinates keys"
        )
    elif isinstance(value, (list, set)):
        # TODO: shouldn't we test value[0][0][0][0] to see if it is MultiPolygon?

        try:
            value[0][0][0]
            return {"$geometry": {"type": "Polygon", "coordinates": value}}
        except (TypeError, IndexError):
            pass

        try:
            value[0][0]
            return {"$geometry": {"type": "LineString", "coordinates": value}}
        except (TypeError, IndexError):
            pass

        try:
            value[0]
            return {"$geometry": {"type": "Point", "coordinates": value}}
        except (TypeError, IndexError):
            pass

    raise InvalidQueryError(
        "Invalid $geometry data. Can be either a "
        "dictionary or (nested) lists of coordinate(s)"
    )


def _prepare_query_for_iterable(field, op, value):
    # We need a special check for BaseDocument, because - although it's iterable - using
    # it as such in the context of this method is most definitely a mistake.
    BaseDocument = _import_class("BaseDocument")

    if isinstance(value, BaseDocument):
        raise TypeError(
            "When using the `in`, `nin`, `all`, or "
            "`near`-operators you can't use a "
            "`Document`, you must wrap your object "
            "in a list (object -> [object])."
        )

    if not hasattr(value, "__iter__"):
        raise TypeError(
            "The `in`, `nin`, `all`, or "
            "`near`-operators must be applied to an "
            "iterable (e.g. a list)."
        )

    return [field.prepare_query_value(op, v) for v in value]
