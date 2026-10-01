"""Static translation of aggregation pipelines from model field names to
database field names.

This module implements the optional ``translate_fields`` mode of
:meth:`mongoengine.queryset.base.BaseQuerySet.aggregate`. When enabled, the
user pipeline may be written with the python attribute names of the model
(including ``db_field``-renamed fields) and mongoengine rewrites every stage
to the actual database field names before sending it to the server.

The translator tracks the shape of the documents as they flow through the
pipeline:

* fields produced by ``$project``/``$addFields``/``$group``/... are tracked so
  that computed fields are never renamed as if they were model fields,
* ``$map``/``$filter``/``$reduce``/``$let`` variable scopes are understood, as
  well as ``$$ROOT``/``$$CURRENT``,
* stages that cannot be analyzed statically (``$lookup``, ``$search``, ...)
  raise :class:`~mongoengine.errors.InvalidQueryError` and must be wrapped in
  :func:`raw` to be passed through unchanged,
* the input pipeline is never mutated in place.
"""

import copy

from mongoengine.common import _import_class
from mongoengine.errors import InvalidQueryError

__all__ = ("Raw", "raw", "translate_pipeline")


class Raw(dict):
    """Marker dict subclass: its content is passed to the server unchanged."""


def raw(stage):
    """Mark a pipeline stage (or a sub-expression) as raw.

    Raw stages are passed through the field-name translation unchanged. Use
    this for stages that cannot be analyzed statically (e.g. ``$lookup``)::

        pipeline = [
            {"$match": {"name": "foo"}},
            raw({"$lookup": {"from": "other", "localField": "n", ...}}),
        ]
    """
    if isinstance(stage, Raw):
        return stage
    if not isinstance(stage, dict):
        raise TypeError(f"raw() expects a dict, got: {type(stage)}")
    return Raw(stage)


class _FieldInfo:
    __slots__ = ("db_name", "subscope")

    def __init__(self, db_name, subscope=None):
        self.db_name = db_name
        self.subscope = subscope


class _Scope:
    """The set of fields available on the documents at a given pipeline stage.

    ``fields`` maps python-side names to :class:`_FieldInfo`. When ``strict``
    is False the shape is unknown (dynamic documents, computed sub-documents)
    and any name resolves to itself.
    """

    __slots__ = ("fields", "strict")

    def __init__(self, fields, strict=True):
        self.fields = fields
        self.strict = strict


# Scope used whenever the shape of a (sub-)document cannot be known statically.
_OPAQUE = _Scope({}, strict=False)


_DOCUMENT_SCOPES = {}


def _subscope_for_field(field):
    ListField = _import_class("ListField")
    EmbeddedDocumentField = _import_class("EmbeddedDocumentField")
    DictField = _import_class("DictField")
    DynamicField = _import_class("DynamicField")
    MapField = _import_class("MapField")
    GenericEmbeddedDocumentField = _import_class("GenericEmbeddedDocumentField")

    if isinstance(field, ListField):
        inner = getattr(field, "field", None)
        if inner is None:
            return None
        return _subscope_for_field(inner)
    if isinstance(field, EmbeddedDocumentField):
        try:
            document_type = field.document_type
        except Exception:
            return _OPAQUE
        if isinstance(document_type, str):
            return _OPAQUE
        return _scope_from_document(document_type)
    opaque_types = (DictField, DynamicField, MapField, GenericEmbeddedDocumentField)
    if isinstance(field, opaque_types):
        return _OPAQUE
    return None


def _scope_from_document(document_cls):
    """Build (and cache) the root scope for a Document/EmbeddedDocument class."""
    scope = _DOCUMENT_SCOPES.get(document_cls)
    if scope is not None:
        return scope
    scope = _Scope({}, strict=not getattr(document_cls, "_dynamic", False))
    # Cache before recursing so self-referencing embedded documents terminate.
    _DOCUMENT_SCOPES[document_cls] = scope
    for name, field in document_cls._fields.items():
        scope.fields[name] = _FieldInfo(field.db_field, _subscope_for_field(field))
    id_field = document_cls._meta.get("id_field")
    if id_field and id_field in scope.fields:
        id_info = scope.fields[id_field]
        scope.fields.setdefault("pk", id_info)
        if id_info.db_name == "_id":
            scope.fields.setdefault("_id", id_info)
    return scope


def _copy_scope(scope):
    if scope is None:
        return None
    return _Scope(
        {
            name: _FieldInfo(info.db_name, _copy_scope(info.subscope))
            for name, info in scope.fields.items()
        },
        strict=scope.strict,
    )


def _resolve_chain(scope, path, strict):
    """Resolve a dotted path against a scope.

    Returns a list of ``(python_name, db_name, _FieldInfo|None)`` tuples, one
    per path part. Unknown parts resolve to themselves (identity) unless the
    scope is strict and ``strict`` resolution was requested, in which case an
    error is raised.
    """
    chain = []
    node = scope
    for part in path.split("."):
        if node is not None and part.isdigit():
            chain.append((part, part, None))
            continue
        info = node.fields.get(part) if node is not None else None
        if info is not None:
            chain.append((part, info.db_name, info))
            node = info.subscope
        else:
            if strict and node is not None and node.strict:
                raise InvalidQueryError(
                    f"Cannot resolve field {part!r} in path {path!r}; if this path "
                    "is produced by a stage that cannot be tracked, wrap that "
                    "stage in raw()"
                )
            chain.append((part, part, None))
            node = None
    return chain


def _resolve_path(scope, path):
    """Translate a field path, raising on unknown fields. Returns (db_path, info)."""
    chain = _resolve_chain(scope, path, strict=True)
    return ".".join(db for _, db, _ in chain), chain[-1][2]


def _translate_output_path(scope, path):
    """Translate as much of an output path as is known, leniently.

    Returns ``(db_path, chain)``. Unknown parts are kept as-is so that new
    (computed) fields can be introduced.
    """
    chain = _resolve_chain(scope, path, strict=False)
    return ".".join(db for _, db, _ in chain), chain


def _plant(fields, chain, leaf_subscope):
    """Insert a resolved chain into a ``fields`` dict of a scope under construction."""
    node = fields
    for py_name, db_name, _ in chain[:-1]:
        info = node.get(py_name)
        if info is None:
            info = _FieldInfo(db_name, _Scope({}, strict=True))
            node[py_name] = info
        elif info.subscope is None:
            info.subscope = _Scope({}, strict=True)
        node = info.subscope.fields
    py_name, db_name, _ = chain[-1]
    node[py_name] = _FieldInfo(db_name, leaf_subscope)


def _plant_id_aliases(fields, subscope):
    info = _FieldInfo("_id", subscope)
    fields["_id"] = info
    fields["id"] = info
    fields["pk"] = info


def _remove_path(scope, path):
    if path in ("_id", "id", "pk"):
        for alias in ("_id", "id", "pk"):
            scope.fields.pop(alias, None)
        return
    parts = path.split(".")
    node = scope
    for part in parts[:-1]:
        if node is None:
            return
        info = node.fields.get(part)
        node = info.subscope if info is not None else None
    if node is not None:
        node.fields.pop(parts[-1], None)


def _is_expression_dict(value):
    return isinstance(value, dict) and any(
        isinstance(key, str) and key.startswith("$") for key in value
    )


def _is_nested_spec(value, key):
    """Whether a $project/$addFields value is a nested specification dict."""
    return (
        isinstance(value, dict) and not _is_expression_dict(value) and "." not in key
    )


def _relative_out_key(chain, key):
    """The output key for a spec entry, relative to its nesting level.

    ``chain`` is the resolved chain of the full (prefixed) path; only the
    trailing parts that correspond to ``key`` itself are kept.
    """
    parts = key.count(".") + 1
    return ".".join(db for _, db, _ in chain[-parts:])


def _translate_expr_string(scope, expr, variables):
    if expr.startswith("$$"):
        var, sep, rest = expr[2:].partition(".")
        if var in ("ROOT", "CURRENT"):
            target = scope
        elif var in variables:
            target = variables[var]
        else:
            # System variables ($$NOW, ...) and unknown variables pass through.
            return expr
        if not sep or target is None:
            return expr
        translated, _ = _resolve_path(target, rest)
        return f"$${var}.{translated}"
    if expr.startswith("$"):
        path = expr[1:]
        if not path:
            return expr
        translated, _ = _resolve_path(scope, path)
        return "$" + translated
    return expr


def _element_scope_of(scope, input_expr):
    """Best-effort scope of the elements produced by a $map/$filter input."""
    if (
        isinstance(input_expr, str)
        and input_expr.startswith("$")
        and not input_expr.startswith("$$")
    ):
        _, info = _resolve_path(scope, input_expr[1:])
        if info is not None:
            return info.subscope
    return None


def _translate_iter_op(scope, spec, variables, result_key):
    """Translate a $map/$filter specification, tracking the ``as`` variable."""
    if not isinstance(spec, dict):
        raise InvalidQueryError(
            f"$map/$filter expect a dict specification, got: {spec!r}"
        )
    element_scope = _element_scope_of(scope, spec.get("input"))
    inner_variables = dict(variables)
    inner_variables[spec.get("as", "this")] = element_scope
    out = {}
    for key, value in spec.items():
        if key == "as":
            out[key] = value
        elif key == result_key:
            out[key] = _translate_expression(scope, value, inner_variables)
        else:
            out[key] = _translate_expression(scope, value, variables)
    return out


def _translate_reduce(scope, spec, variables):
    if not isinstance(spec, dict):
        raise InvalidQueryError(f"$reduce expects a dict specification, got: {spec!r}")
    element_scope = _element_scope_of(scope, spec.get("input"))
    inner_variables = dict(variables)
    inner_variables["this"] = element_scope
    inner_variables["value"] = None
    out = {}
    for key, value in spec.items():
        if key == "in":
            out[key] = _translate_expression(scope, value, inner_variables)
        else:
            out[key] = _translate_expression(scope, value, variables)
    return out


def _translate_let(scope, spec, variables):
    if not isinstance(spec, dict):
        raise InvalidQueryError(f"$let expects a dict specification, got: {spec!r}")
    inner_variables = dict(variables)
    translated_vars = {}
    for name, value in (spec.get("vars") or {}).items():
        translated_vars[name] = _translate_expression(scope, value, variables)
        # The shape of a bound variable cannot be tracked: keep it opaque.
        inner_variables[name] = None
    out = {}
    for key, value in spec.items():
        if key == "vars":
            out[key] = translated_vars
        elif key == "in":
            out[key] = _translate_expression(scope, value, inner_variables)
        else:
            out[key] = _translate_expression(scope, value, variables)
    return out


def _translate_expression(scope, expr, variables):
    if isinstance(expr, Raw):
        return dict(expr)
    if isinstance(expr, str):
        return _translate_expr_string(scope, expr, variables)
    if isinstance(expr, (list, tuple)):
        return [_translate_expression(scope, item, variables) for item in expr]
    if isinstance(expr, dict):
        if "$literal" in expr:
            return dict(expr)
        if "$map" in expr:
            return {"$map": _translate_iter_op(scope, expr["$map"], variables, "in")}
        if "$filter" in expr:
            filter_spec = _translate_iter_op(scope, expr["$filter"], variables, "cond")
            return {"$filter": filter_spec}
        if "$reduce" in expr:
            return {"$reduce": _translate_reduce(scope, expr["$reduce"], variables)}
        if "$let" in expr:
            return {"$let": _translate_let(scope, expr["$let"], variables)}
        return {
            key: _translate_expression(scope, value, variables)
            for key, value in expr.items()
        }
    return expr


def _translate_match_value(subscope, value):
    if isinstance(value, Raw):
        return dict(value)
    if isinstance(value, dict):
        if _is_expression_dict(value):
            out = {}
            for operator, operand in value.items():
                if operator == "$elemMatch":
                    out[operator] = _translate_match(
                        subscope if subscope is not None else _OPAQUE, operand
                    )
                elif operator == "$not":
                    out[operator] = _translate_match_value(subscope, operand)
                else:
                    out[operator] = operand
            return out
        if subscope is not None:
            # Literal document matched against a known embedded document.
            return _translate_match(subscope, value)
    return value


def _translate_match(scope, spec):
    if isinstance(spec, Raw):
        return dict(spec)
    if not isinstance(spec, dict):
        raise InvalidQueryError(f"$match expects a dict specification, got: {spec!r}")
    out = {}
    for key, value in spec.items():
        if key in ("$and", "$or", "$nor"):
            if not isinstance(value, (list, tuple)):
                raise InvalidQueryError(f"{key} expects a list of sub-expressions")
            out[key] = [_translate_match(scope, item) for item in value]
        elif key == "$expr":
            out[key] = _translate_expression(scope, value, {})
        elif key in ("$jsonSchema", "$text", "$where", "$comment"):
            out[key] = value
        elif key.startswith("$"):
            raise InvalidQueryError(
                f"Cannot statically translate operator {key!r} in a $match stage; "
                "wrap the stage in raw() to pass it through unchanged"
            )
        else:
            translated, info = _resolve_path(scope, key)
            subscope = info.subscope if info is not None else None
            out[translated] = _translate_match_value(subscope, value)
    return out


def _stage_match(scope, spec):
    return _translate_match(scope, spec), scope


def _stage_sort(scope, spec):
    if not isinstance(spec, dict):
        raise InvalidQueryError(f"$sort expects a dict of field paths, got: {spec!r}")
    out = {}
    for key, value in spec.items():
        translated, _ = _resolve_path(scope, key)
        out[translated] = value
    return out, scope


def _project_rec(scope, spec, prefix, out, new_fields, exclusions):
    """Walk a $project specification.

    ``scope`` is always the *input* document scope (expressions are absolute),
    ``prefix`` is the dotted python-side output path of the nested spec being
    processed. Returns True if any inclusion/computed field was seen.
    """
    saw_inclusion = False
    for key, value in spec.items():
        full_path = prefix + key
        if _is_nested_spec(value, key):
            sub_out = {}
            sub_saw = _project_rec(
                scope, value, full_path + ".", sub_out, new_fields, exclusions
            )
            _, chain = _translate_output_path(scope, full_path)
            out[chain[-1][1]] = sub_out
            saw_inclusion = saw_inclusion or sub_saw
        elif isinstance(value, int):  # 0/1/True/False inclusion flags
            if value:
                saw_inclusion = saw_inclusion or full_path != "_id"
                chain = _resolve_chain(scope, full_path, strict=True)
                info = chain[-1][2]
                _plant(new_fields, chain, info.subscope if info is not None else None)
                if full_path == "_id":
                    subscope = info.subscope if info is not None else None
                    _plant_id_aliases(new_fields, subscope)
            else:
                _, chain = _translate_output_path(scope, full_path)
                exclusions.append(full_path)
            out[_relative_out_key(chain, key)] = value
        else:
            saw_inclusion = saw_inclusion or full_path != "_id"
            _, chain = _translate_output_path(scope, full_path)
            out[_relative_out_key(chain, key)] = _translate_expression(scope, value, {})
            _plant(new_fields, chain, None)
            if full_path == "_id":
                _plant_id_aliases(new_fields, None)
    return saw_inclusion


def _stage_project(scope, spec):
    if not isinstance(spec, dict):
        raise InvalidQueryError(f"$project expects a dict specification, got: {spec!r}")
    out = {}
    new_fields = {}
    exclusions = []
    saw_inclusion = _project_rec(scope, spec, "", out, new_fields, exclusions)
    if saw_inclusion:
        if "_id" not in spec:
            for alias in ("_id", "id", "pk"):
                if alias in scope.fields:
                    new_fields[alias] = scope.fields[alias]
        new_scope = _Scope(new_fields, strict=True)
    else:
        new_scope = _copy_scope(scope)
    for path in exclusions:
        _remove_path(new_scope, path)
    return out, new_scope


def _add_fields_rec(scope, spec, prefix, new_scope, out):
    for key, value in spec.items():
        full_path = prefix + key
        _, chain = _translate_output_path(scope, full_path)
        if _is_nested_spec(value, key):
            sub_out = {}
            _add_fields_rec(scope, value, full_path + ".", new_scope, sub_out)
            out[chain[-1][1]] = sub_out
        else:
            out[_relative_out_key(chain, key)] = _translate_expression(scope, value, {})
            _plant(new_scope.fields, chain, None)


def _stage_add_fields(scope, spec):
    if not isinstance(spec, dict):
        raise InvalidQueryError(
            f"$addFields/$set expect a dict specification, got: {spec!r}"
        )
    new_scope = _copy_scope(scope)
    out = {}
    _add_fields_rec(scope, spec, "", new_scope, out)
    return out, new_scope


def _stage_unset(scope, spec):
    if isinstance(spec, str):
        paths, single = [spec], True
    elif isinstance(spec, (list, tuple)):
        paths, single = list(spec), False
    else:
        raise InvalidQueryError(
            f"$unset expects a field path or a list of paths, got: {spec!r}"
        )
    new_scope = _copy_scope(scope)
    out_paths = []
    for path in paths:
        chain = _resolve_chain(scope, path, strict=True)
        out_paths.append(".".join(db for _, db, _ in chain))
        _remove_path(new_scope, path)
    return (out_paths[0] if single else out_paths), new_scope


def _stage_group(scope, spec):
    if not isinstance(spec, dict) or "_id" not in spec:
        raise InvalidQueryError("$group expects a dict specification containing '_id'")
    out = {}
    new_fields = {"_id": _FieldInfo("_id", None)}
    for key, value in spec.items():
        out[key] = _translate_expression(scope, value, {})
        if key != "_id":
            new_fields[key] = _FieldInfo(key, None)
    return out, _Scope(new_fields, strict=True)


def _stage_unwind(scope, spec):
    if isinstance(spec, str):
        path, options = spec, None
    elif isinstance(spec, dict):
        options = dict(spec)
        path = options.get("path")
    else:
        raise InvalidQueryError(
            f"$unwind expects a path string or an options dict, got: {spec!r}"
        )
    if not isinstance(path, str) or not path.startswith("$"):
        raise InvalidQueryError(
            f"$unwind path must be a field path starting with '$', got: {path!r}"
        )
    translated, _ = _resolve_path(scope, path[1:])
    if options is None:
        return "$" + translated, scope
    options["path"] = "$" + translated
    new_scope = scope
    index_name = options.get("includeArrayIndex")
    if isinstance(index_name, str):
        new_scope = _copy_scope(scope)
        new_scope.fields[index_name] = _FieldInfo(index_name, None)
    return options, new_scope


def _constructor_scope(scope, expr):
    """Translate an object-constructor expression and infer the shape it produces."""
    out = {}
    fields = {}
    for key, value in expr.items():
        if isinstance(value, dict) and not _is_expression_dict(value):
            sub_out, sub_scope = _constructor_scope(scope, value)
            out[key] = sub_out
            fields[key] = _FieldInfo(key, sub_scope)
        else:
            out[key] = _translate_expression(scope, value, {})
            fields[key] = _FieldInfo(key, None)
    return out, _Scope(fields, strict=True)


def _replace_with_expr(scope, expr):
    if isinstance(expr, str) and expr.startswith("$") and not expr.startswith("$$"):
        translated, info = _resolve_path(scope, expr[1:])
        if info is not None and info.subscope is not None:
            new_scope = info.subscope
        else:
            new_scope = _OPAQUE
        return "$" + translated, new_scope
    if isinstance(expr, dict) and not _is_expression_dict(expr):
        return _constructor_scope(scope, expr)
    raise InvalidQueryError(
        "Cannot statically determine the document shape produced by "
        "$replaceRoot/$replaceWith; wrap the stage in raw() to pass it through "
        "unchanged"
    )


def _stage_replace_root(scope, spec):
    if isinstance(spec, dict) and "newRoot" in spec:
        new_root, new_scope = _replace_with_expr(scope, spec["newRoot"])
        return {"newRoot": new_root}, new_scope
    raise InvalidQueryError("$replaceRoot expects a dict with a 'newRoot' key")


def _stage_replace_with(scope, spec):
    return _replace_with_expr(scope, spec)


def _stage_count(scope, spec):
    if not isinstance(spec, str):
        raise InvalidQueryError(f"$count expects a string field name, got: {spec!r}")
    return spec, _Scope({spec: _FieldInfo(spec, None)}, strict=True)


def _stage_sort_by_count(scope, spec):
    out = _translate_expression(scope, spec, {})
    new_scope = _Scope(
        {"_id": _FieldInfo("_id", None), "count": _FieldInfo("count", None)},
        strict=True,
    )
    return out, new_scope


def _stage_facet(scope, spec):
    if not isinstance(spec, dict):
        raise InvalidQueryError(
            f"$facet expects a dict of sub-pipelines, got: {spec!r}"
        )
    out = {}
    new_fields = {}
    for name, sub_pipeline in spec.items():
        out[name] = _translate_stage_list(sub_pipeline, scope)
        new_fields[name] = _FieldInfo(name, _OPAQUE)
    return out, _Scope(new_fields, strict=True)


def _stage_bucket(scope, spec):
    if not isinstance(spec, dict) or "groupBy" not in spec:
        raise InvalidQueryError(
            "$bucket/$bucketAuto expect a dict specification with 'groupBy'"
        )
    out = dict(spec)
    out["groupBy"] = _translate_expression(scope, spec["groupBy"], {})
    new_fields = {"_id": _FieldInfo("_id", None), "count": _FieldInfo("count", None)}
    output = spec.get("output")
    if isinstance(output, dict):
        translated_output = {}
        for key, value in output.items():
            translated_output[key] = _translate_expression(scope, value, {})
            new_fields[key] = _FieldInfo(key, None)
        out["output"] = translated_output
    return out, _Scope(new_fields, strict=True)


def _stage_redact(scope, spec):
    return _translate_expression(scope, spec, {}), scope


def _stage_geo_near(scope, spec):
    new_scope = _copy_scope(scope)
    if isinstance(spec, dict):
        for option in ("distanceField", "includeLocs"):
            name = spec.get(option)
            if isinstance(name, str):
                new_scope.fields[name] = _FieldInfo(name, None)
    return spec, new_scope


def _stage_passthrough(scope, spec):
    return spec, scope


_STAGE_HANDLERS = {
    "$match": _stage_match,
    "$project": _stage_project,
    "$addFields": _stage_add_fields,
    "$set": _stage_add_fields,
    "$unset": _stage_unset,
    "$group": _stage_group,
    "$sort": _stage_sort,
    "$unwind": _stage_unwind,
    "$replaceRoot": _stage_replace_root,
    "$replaceWith": _stage_replace_with,
    "$count": _stage_count,
    "$sortByCount": _stage_sort_by_count,
    "$facet": _stage_facet,
    "$bucket": _stage_bucket,
    "$bucketAuto": _stage_bucket,
    "$redact": _stage_redact,
    "$geoNear": _stage_geo_near,
    "$limit": _stage_passthrough,
    "$skip": _stage_passthrough,
    "$sample": _stage_passthrough,
    "$out": _stage_passthrough,
    "$merge": _stage_passthrough,
    "$collStats": _stage_passthrough,
    "$indexStats": _stage_passthrough,
}


def _translate_stage_list(pipeline, scope):
    if not isinstance(pipeline, (list, tuple)):
        raise InvalidQueryError(
            f"Aggregation pipeline must be a list of stage dicts, got: {pipeline!r}"
        )
    out = []
    for stage in pipeline:
        if isinstance(stage, Raw):
            out.append(dict(stage))
            continue
        if not isinstance(stage, dict) or len(stage) != 1:
            raise InvalidQueryError(
                "Each aggregation pipeline stage must be a single-key dict, "
                f"got: {stage!r}"
            )
        (operator, spec), = stage.items()
        handler = _STAGE_HANDLERS.get(operator)
        if handler is None:
            raise InvalidQueryError(
                f"Aggregation stage {operator!r} cannot be statically translated; "
                "wrap it in raw() to pass it through unchanged"
            )
        new_spec, scope = handler(scope, spec)
        out.append({operator: new_spec})
    return out


def translate_pipeline(document_cls, pipeline):
    """Translate a pipeline written with model field names to database field names.

    :param document_cls: the Document class the pipeline runs against
    :param pipeline: list of aggregation stages using python attribute names

    Returns a new pipeline; the input is never mutated. Stages that cannot be
    analyzed statically must be wrapped in :func:`raw`.
    """
    scope = _scope_from_document(document_cls)
    return _translate_stage_list(copy.deepcopy(pipeline), scope)
