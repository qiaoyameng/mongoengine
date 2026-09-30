import re
from collections import defaultdict

import pymongo
from bson import ObjectId
from bson.dbref import DBRef

from mongoengine.base import UPDATE_OPERATORS
from mongoengine.common import _import_class
from mongoengine.errors import InvalidQueryError

__all__ = ("query", "update", "array_filters", "STRING_OPERATORS")

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
                        # Convert the filtered positional operator
                        # S<identifier> to $[<identifier>]
                        identifier = _array_filter_identifier(field)
                        if identifier is not None:
                            field = "$[%s]" % identifier
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


# A filtered positional identifier is written as ``S<name>`` in django-style
# update paths (e.g. ``set__tags__Selem__name``) and rendered as
# ``$[elem].name``. ``S`` on its own is the plain positional operator ``$``.
_ARRAY_FILTER_IDENTIFIER_RE = re.compile(r"^S(?P<name>[A-Za-z][A-Za-z0-9]*)$")


def _array_filter_identifier(part):
    """Return the identifier declared by an update path part or ``None``."""
    match = _ARRAY_FILTER_IDENTIFIER_RE.match(part)
    if match:
        return match.group("name")
    return None


def _list_element_type(field):
    """Return the field type stored by a list-like field or ``None``."""
    ListField = _import_class("ListField")
    EmbeddedDocumentListField = _import_class("EmbeddedDocumentListField")
    if isinstance(field, (ListField, EmbeddedDocumentListField)):
        return field.field
    return None


def _is_same_element_type(first, second):
    """Return whether two list element fields are type-compatible."""
    if type(first) is not type(second):
        return False
    EmbeddedDocumentField = _import_class("EmbeddedDocumentField")
    if isinstance(first, EmbeddedDocumentField):
        return first.document_type is second.document_type
    return True


def _element_type_label(field):
    """Human readable description of a list element field."""
    EmbeddedDocumentField = _import_class("EmbeddedDocumentField")
    if isinstance(field, EmbeddedDocumentField):
        return "%s(%s)" % (
            type(field).__name__,
            field.document_type._class_name,
        )
    return type(field).__name__


def _resolve_update_parts(key):
    """Strip the leading operator/trailing comparison from an update key."""
    parts = key.split("__")

    if len(parts) < 3 and parts[0] not in UPDATE_OPERATORS:
        parts.insert(0, "set")
    if parts[0] in UPDATE_OPERATORS:
        parts = parts[1:]

    if len(parts) > 1 and parts[-1] in COMPARISON_OPERATORS:
        parts = parts[:-1]
    elif len(parts) > 1 and parts[-1] == "":
        parts = parts[:-1]

    return parts


def _collect_array_filter_bindings(doc_cls, update):
    """Bind each filtered positional identifier to its list element field.

    The element field is inferred from the list field preceding the
    placeholder in the update path. An identifier bound to conflicting
    element types raises :class:`~mongoengine.errors.InvalidQueryError`.
    """
    bindings = {}

    for key in update:
        if key == "__raw__":
            continue

        path_parts = _resolve_update_parts(key)
        leading_operator = key.split("__", 1)[0]
        if leading_operator in UPDATE_OPERATORS:
            leading_operator = {
                "push_all": "pushAll",
                "pull_all": "pullAll",
                "dec": "inc",
                "add_to_set": "addToSet",
                "set_on_insert": "setOnInsert",
            }.get(leading_operator, leading_operator)
        else:
            leading_operator = "set"

        current_field = None
        for part in path_parts:
            if part.isdigit():
                continue

            identifier = _array_filter_identifier(part)
            # A declared field takes precedence over the placeholder syntax,
            # so only attempt resolution when the part is not a placeholder
            # or might shadow one as a real field name.
            next_field = None
            if identifier is None or current_field is None:
                if current_field is None:
                    lookup_name = part
                    if lookup_name == "pk":
                        lookup_name = doc_cls._meta["id_field"]
                    next_field = doc_cls._fields.get(lookup_name)
                else:
                    inner_field = getattr(current_field, "field", None)
                    if inner_field is not None and hasattr(
                        inner_field, "lookup_member"
                    ):
                        next_field = inner_field.lookup_member(part)
                    elif hasattr(current_field, "lookup_member"):
                        next_field = current_field.lookup_member(part)

                if next_field is not None:
                    identifier = None

            if identifier is not None:
                if leading_operator in ("pull", "pullAll"):
                    raise InvalidQueryError(
                        "Array filter placeholders cannot be used with the "
                        "'%s' update operator" % leading_operator
                    )

                element_field = _list_element_type(current_field)
                if element_field is None:
                    raise InvalidQueryError(
                        "Array filter identifier '%s' in update path '%s' "
                        "must immediately follow a ListField"
                        % (identifier, key)
                    )

                existing = bindings.get(identifier)
                if existing is not None and not _is_same_element_type(
                    existing, element_field
                ):
                    raise InvalidQueryError(
                        "Array filter identifier '%s' is bound to conflicting "
                        "list element types: %s and %s"
                        % (
                            identifier,
                            _element_type_label(existing),
                            _element_type_label(element_field),
                        )
                    )
                bindings[identifier] = element_field
                # The placeholder stands for the list element.
                current_field = element_field
                continue

            if next_field is None:
                # Unresolved subfield: stop traversing this path. Later
                # update validation reports any genuinely invalid path.
                break

            current_field = next_field

    return bindings


def _prepare_scalar_filter_value(field, value):
    """Convert a scalar array filter value, transparently keeping operators."""
    if isinstance(value, dict) and value and all(
        isinstance(k, str) and k.startswith("$") for k in value
    ):
        return {
            operator: _prepare_scalar_filter_operand(field, operator, operand)
            for operator, operand in value.items()
        }
    return field.prepare_query_value(None, value)


def _prepare_scalar_filter_operand(field, operator, value):
    name = operator[1:]
    if name in ("in", "nin", "all"):
        return _prepare_query_for_iterable(field, name, value)
    if name in COMPARISON_OPERATORS or name in STRING_OPERATORS:
        return field.prepare_query_value(name, value)
    return value


def _convert_array_filter_expression(identifier, element_field, suffix, value):
    """Convert one key/value pair of a model-aware array filter."""
    EmbeddedDocumentField = _import_class("EmbeddedDocumentField")

    if suffix:
        if isinstance(element_field, EmbeddedDocumentField):
            converted = query(
                element_field.document_type, **{"__".join(suffix): value}
            )
            sub_key, sub_value = next(iter(converted.items()))
            return "%s.%s" % (identifier, sub_key), sub_value

        # Scalar list element: the suffix must be a comparison operator
        operator = suffix[0]
        if operator not in COMPARISON_OPERATORS and operator not in STRING_OPERATORS:
            raise InvalidQueryError(
                "Array filter '%s__%s' is invalid: '%s' is not an embedded "
                "document field nor a query operator"
                % (identifier, "__".join(suffix), operator)
            )
        operand = _prepare_scalar_filter_operand(
            element_field, "$" + operator, value
        )
        return identifier, {"$" + operator: operand}

    if isinstance(element_field, EmbeddedDocumentField):
        return identifier, element_field.prepare_query_value(None, value)
    return identifier, _prepare_scalar_filter_value(element_field, value)


def array_filters(doc_cls, filters, update):
    """Convert model-aware array filters into the MongoDB ``arrayFilters``.

    :param doc_cls: document class the update is performed on
    :param filters: list of dictionaries keyed by the filtered positional
        identifiers used in the update paths (``elem`` /
        ``elem__embedded_field``), with python field names and values
    :param update: django-style update keyword arguments used to bind the
        identifiers to the corresponding list element types

    Field aliases (``db_field``) and python values are converted like regular
    queries, including nested ``$elemMatch`` expressions on several levels of
    embedded lists. Undeclared, unused or inconsistently typed identifiers
    raise :class:`~mongoengine.errors.InvalidQueryError` before any request is
    sent to the database.
    """
    if filters is None:
        return None
    if not isinstance(filters, (list, tuple)):
        raise InvalidQueryError("array_filters must be a list of filters")

    bindings = _collect_array_filter_bindings(doc_cls, update)

    converted_filters = []
    used_identifiers = set()

    for filter_expr in filters:
        if not isinstance(filter_expr, dict):
            raise InvalidQueryError("Each array filter must be a dictionary")

        converted_expr = {}
        for key, value in filter_expr.items():
            parts = key.split("__")
            identifier = parts[0]
            suffix = parts[1:]

            if identifier not in bindings:
                raise InvalidQueryError(
                    "Array filter identifier '%s' is not declared in any "
                    "update path (use an 'S%s' placeholder)"
                    % (identifier, identifier)
                )
            used_identifiers.add(identifier)

            converted_key, converted_value = _convert_array_filter_expression(
                identifier, bindings[identifier], suffix, value
            )
            if converted_key in converted_expr:
                existing_value = converted_expr[converted_key]
                if isinstance(existing_value, dict) and isinstance(
                    converted_value, dict
                ):
                    for operator in converted_value:
                        if operator in existing_value:
                            raise InvalidQueryError(
                                "Conflicting array filter conditions on '%s' "
                                "for operator '%s'"
                                % (converted_key, operator)
                            )
                    existing_value.update(converted_value)
                else:
                    raise InvalidQueryError(
                        "Conflicting array filter conditions on '%s'"
                        % converted_key
                    )
            else:
                converted_expr[converted_key] = converted_value

        converted_filters.append(converted_expr)

    unused_identifiers = set(bindings) - used_identifiers
    if unused_identifiers:
        raise InvalidQueryError(
            "The array filter identifier(s) %s are declared in update paths "
            "but no matching array filter was provided"
            % ", ".join(sorted(unused_identifiers))
        )

    return converted_filters
