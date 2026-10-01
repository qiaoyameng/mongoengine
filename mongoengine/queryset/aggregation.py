import copy
from collections.abc import Mapping

from bson import encode

from mongoengine.common import _import_class
from mongoengine.errors import InvalidQueryError


RAW = "$raw"
_ROOT_VARIABLES = {"$$DESCEND", "$$KEEP", "$$PRUNE"}
_SYSTEM_VARIABLES = {
    "$$NOW",
    "$$CLUSTER_TIME",
    "$$REMOVE",
}


class _Node:
    __slots__ = ("document_type", "db_name", "children", "item", "dict_field", "dynamic", "opaque")

    def __init__(
        self,
        document_type=None,
        db_name=None,
        children=None,
        item=None,
        dict_field=False,
        dynamic=False,
        opaque=False,
    ):
        self.document_type = document_type
        self.db_name = db_name
        self.children = children if children is not None else {}
        self.item = item
        self.dict_field = dict_field
        self.dynamic = dynamic
        self.opaque = opaque


def _opaque(db_name=None):
    return _Node(db_name=db_name, opaque=True)


def _model_node(document_type, db_name=None):
    ListField = _import_class("ListField")
    DictField = _import_class("DictField")
    EmbeddedDocumentField = _import_class("EmbeddedDocumentField")
    GenericEmbeddedDocumentField = _import_class("GenericEmbeddedDocumentField")
    DynamicField = _import_class("DynamicField")

    children = {}
    node = _Node(document_type=document_type, db_name=db_name, children=children, dynamic=bool(getattr(document_type, "_dynamic", False)))
    for name, field in document_type._fields.items():
        if isinstance(field, ListField):
            if isinstance(field.field, EmbeddedDocumentField):
                item = _model_node(field.field.document_type)
                item.db_name = None
            elif isinstance(field.field, GenericEmbeddedDocumentField):
                item = _opaque()
            elif field.field is not None:
                item = _node_for_field(field.field)
            else:
                item = _opaque()
            children[name] = _Node(db_name=field.db_field, item=item)
        elif isinstance(field, EmbeddedDocumentField):
            children[name] = _model_node(field.document_type, db_name=field.db_field)
        elif isinstance(field, (GenericEmbeddedDocumentField, DynamicField)):
            children[name] = _opaque(db_name=field.db_field)
        elif isinstance(field, DictField):
            children[name] = _Node(
                db_name=field.db_field,
                dict_field=True,
                item=_node_for_field(field.field) if field.field is not None else _opaque(),
            )
        else:
            children[name] = _Node(db_name=field.db_field)

    if getattr(document_type, "_meta", {}).get("allow_inheritance", False):
        children["_cls"] = _Node(db_name="_cls")
    return node


def _node_for_field(field):
    EmbeddedDocumentField = _import_class("EmbeddedDocumentField")
    ListField = _import_class("ListField")
    DictField = _import_class("DictField")
    if isinstance(field, ListField):
        if isinstance(field.field, EmbeddedDocumentField):
            return _Node(db_name=field.db_field, item=_model_node(field.field.document_type))
        return _Node(db_name=field.db_field, item=_opaque())
    if isinstance(field, EmbeddedDocumentField):
        return _model_node(field.document_type, db_name=field.db_field)
    if isinstance(field, DictField):
        return _Node(
            db_name=field.db_field,
            dict_field=True,
            item=_node_for_field(field.field) if field.field is not None else _opaque(),
        )
    return _Node(db_name=field.db_field)


def _clone(node):
    if node is None:
        return None
    return _Node(
        document_type=node.document_type,
        db_name=node.db_name,
        children={key: _clone(value) for key, value in node.children.items()},
        item=_clone(node.item),
        dict_field=node.dict_field,
        dynamic=node.dynamic,
        opaque=node.opaque,
    )


def _synthetic(node):
    cloned = _clone(node)
    _strip_document_type(cloned)
    return cloned


def _strip_document_type(node):
    if node is None:
        return
    node.document_type = None
    node.dynamic = False
    for child in node.children.values():
        _strip_document_type(child)
    if node.item is not None:
        _strip_document_type(node.item)


def _physical_name(node, logical):
    if node.document_type is not None and logical in node.document_type._fields:
        return node.document_type._fields[logical].db_field
    return logical


def _path_error(path, stage):
    return InvalidQueryError(
        f"Cannot statically resolve field {path!r} for aggregate {stage!r}; use $raw"
    )


def _resolve_path(node, path, stage):
    if node is None or node.opaque:
        return path, node

    parts = str(path).split(".")
    output = []
    current = node
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if current is None or current.opaque:
            output.append(part)
            continue
        if current.dict_field:
            output.append(part)
            current = current.item if current.item is not None else _opaque()
            continue
        if current.item is not None:
            if not last:
                output.append(current.db_name or part)
                current = current.item
                continue
            elif part.isdigit():
                output.append(part)
                current = current.item
            else:
                output.append(current.db_name or part)
            continue
        child = current.children.get(part)
        if child is not None:
            if child.item is not None and not child.dict_field and not last:
                output.append(child.db_name or part)
                current = child.item
                continue
            output.append(child.db_name or part)
            current = child
            continue
        if current.document_type is None and current.item is None and current.children:
            raise _path_error(path, stage)
        if current.document_type is not None:
            field = current.document_type._fields.get(part)
            if field is None and part in ("id", "pk"):
                part = current.document_type._meta["id_field"]
                field = current.document_type._fields.get(part)
            if field is not None:
                output.append(field.db_field)
                if last:
                    current = current.children.get(part, _node_for_field(field))
                else:
                    current = current.children.get(part, _node_for_field(field))
                continue
            if current.dynamic:
                output.append(part)
                current = _opaque()
                continue
            raise _path_error(path, stage)
        raise _path_error(path, stage)
    return ".".join(output), current


def _remove_path(root, path):
    parts = str(path).split(".")
    current = root
    for index, part in enumerate(parts[:-1]):
        if current is None or current.opaque:
            return
        if current.item is not None:
            if part.isdigit():
                current = current.item
                continue
            current = current.item
            continue
        if current.document_type is not None:
            field = current.document_type._fields.get(part)
            if field is None and part in ("id", "pk"):
                part = current.document_type._meta["id_field"]
                field = current.document_type._fields.get(part)
            current = current.children.get(part) if field else None
        else:
            current = current.children.get(part)
        if current is None:
            return
    if current is None or current.opaque or current.item is not None:
        return
    part = parts[-1]
    if current.document_type is not None:
        field = current.document_type._fields.get(part)
        if field is None and part in ("id", "pk"):
            part = current.document_type._meta["id_field"]
            field = current.document_type._fields.get(part)
        if field is not None:
            current.children.pop(part, None)
    else:
        current.children.pop(part, None)


class _Scope:
    __slots__ = ("root", "variables", "parent")

    def __init__(self, root, variables=None, parent=None):
        self.root = root
        self.variables = variables or {}
        self.parent = parent

    def find(self, name):
        if name in self.variables:
            return self.variables[name]
        if self.parent is not None:
            return self.parent.find(name)
        return None


def _convert_expression(value, root, stage, scope=None):
    if scope is None:
        scope = _Scope(root)

    if isinstance(value, Mapping):
        if RAW in value:
            if len(value) != 1:
                raise InvalidQueryError("$raw cannot be combined with other expression keys")
            return copy.deepcopy(value[RAW]), _opaque()

        operator_keys = [key for key in value if isinstance(key, str) and key.startswith("$")]
        if not operator_keys:
            result = {}
            for key, item in value.items():
                converted, _ = _convert_expression(item, root, stage, scope)
                result[key] = converted
            return result, _opaque()

        result = {}
        last_node = _opaque()
        for key, item in value.items():
            if key == "$literal":
                result[key] = copy.deepcopy(item)
                last_node = _opaque()
            elif key == "$map":
                converted, last_node = _convert_map(item, root, stage, scope)
                result[key] = converted
            elif key == "$filter":
                converted, last_node = _convert_filter(item, root, stage, scope)
                result[key] = converted
            elif key == "$reduce":
                converted, last_node = _convert_reduce(item, root, stage, scope)
                result[key] = converted
            elif key == "$let":
                converted, last_node = _convert_let(item, root, stage, scope)
                result[key] = converted
            elif key == "$arrayElemAt" and isinstance(item, list) and item:
                converted_array, array_node = _convert_expression(
                    item[0], root, stage, scope
                )
                converted_items = [converted_array]
                for entry in item[1:]:
                    converted, _ = _convert_expression(entry, root, stage, scope)
                    converted_items.append(converted)
                result[key] = converted_items
                last_node = (
                    array_node.item
                    if array_node is not None and array_node.item is not None
                    else _opaque()
                )
            elif key in ("$getField", "$setField", "$unsetField"):
                raise InvalidQueryError(f"Use $raw for {key} in aggregate {stage!r}")
            elif isinstance(item, list):
                converted_items = []
                for entry in item:
                    converted, node = _convert_expression(entry, root, stage, scope)
                    converted_items.append(converted)
                    last_node = node
                result[key] = converted_items
            else:
                result[key], last_node = _convert_expression(item, root, stage, scope)
        return result, last_node

    if isinstance(value, list):
        result = []
        last_node = _opaque()
        for item in value:
            converted, node = _convert_expression(item, root, stage, scope)
            result.append(converted)
            last_node = node
        return result, _Node(item=last_node)

    if isinstance(value, str) and value.startswith("$"):
        return _convert_field_string(value, root, stage, scope)

    return copy.deepcopy(value), None


def _convert_map(spec, root, stage, parent_scope):
    if not isinstance(spec, Mapping) or not {"input", "in"}.issubset(spec):
        raise InvalidQueryError(f"Invalid $map expression in aggregate {stage!r}")

    converted_input, input_node = _convert_expression(spec["input"], root, stage, parent_scope)
    element_node = input_node.item if input_node is not None and input_node.item is not None else _opaque()
    as_name = spec.get("as", "$$this")
    variable = as_name[2:] if as_name.startswith("$$") else as_name
    scope = _Scope(
        root,
        {
            f"$${variable}": element_node,
            "$$CURRENT": element_node,
            "$$this": element_node,
        },
        parent_scope,
    )
    converted_in, output_node = _convert_expression(spec["in"], root, stage, scope)
    result = {
        "input": converted_input,
        "in": converted_in,
    }
    if "as" in spec:
        result["as"] = as_name
    return result, _Node(item=output_node)


def _convert_filter(spec, root, stage, parent_scope):
    if not isinstance(spec, Mapping) or not {"input", "cond"}.issubset(spec):
        raise InvalidQueryError(f"Invalid $filter expression in aggregate {stage!r}")

    converted_input, input_node = _convert_expression(spec["input"], root, stage, parent_scope)
    element_node = input_node.item if input_node is not None and input_node.item is not None else _opaque()
    as_name = spec.get("as", "$$this")
    variable = as_name[2:] if as_name.startswith("$$") else as_name
    scope = _Scope(
        root,
        {
            f"$${variable}": element_node,
            "$$CURRENT": element_node,
            "$$this": element_node,
        },
        parent_scope,
    )
    converted_cond, _ = _convert_expression(spec["cond"], root, stage, scope)
    result = {
        "input": converted_input,
        "cond": converted_cond,
    }
    if "as" in spec:
        result["as"] = as_name
    return result, _Node(item=element_node)


def _convert_reduce(spec, root, stage, parent_scope):
    if not isinstance(spec, Mapping) or not {"input", "initialValue", "in"}.issubset(spec):
        raise InvalidQueryError(f"Invalid $reduce expression in aggregate {stage!r}")

    converted_input, input_node = _convert_expression(spec["input"], root, stage, parent_scope)
    converted_initial, _ = _convert_expression(spec["initialValue"], root, stage, parent_scope)
    element_node = input_node.item if input_node is not None and input_node.item is not None else _opaque()
    scope = _Scope(
        root,
        {"$$value": _opaque(), "$$this": element_node, "$$CURRENT": element_node},
        parent_scope,
    )
    converted_in, output_node = _convert_expression(spec["in"], root, stage, scope)
    return {
        "input": converted_input,
        "initialValue": converted_initial,
        "in": converted_in,
    }, output_node


def _convert_let(spec, root, stage, parent_scope):
    if not isinstance(spec, Mapping) or not {"vars", "in"}.issubset(spec):
        raise InvalidQueryError(f"Invalid $let expression in aggregate {stage!r}")
    if not isinstance(spec["vars"], Mapping):
        raise InvalidQueryError(f"Invalid $let vars in aggregate {stage!r}")

    variables = {}
    converted_vars = {}
    for name, expression in spec["vars"].items():
        converted, node = _convert_expression(expression, root, stage, parent_scope)
        variable = name if str(name).startswith("$") else f"${name}"
        converted_vars[name] = converted
        variables[variable] = node or _opaque()

    scope = _Scope(root, variables, parent_scope)
    converted_in, node = _convert_expression(spec["in"], root, stage, scope)
    return {"vars": converted_vars, "in": converted_in}, node


def _convert_field_string(value, root, stage, scope):
    if value in ("$$ROOT", "$$CURRENT"):
        return value, root
    if value in _ROOT_VARIABLES or value in _SYSTEM_VARIABLES:
        return value, _opaque()
    if value.startswith("$$"):
        variable_name, _, path = value[2:].partition(".")
        variable = f"$${variable_name}"
        if variable in ("$$ROOT", "$$CURRENT"):
            node = root
        else:
            node = scope.find(variable)
        if node is None:
            raise InvalidQueryError(f"Unknown aggregation variable {variable!r} in {stage!r}; use $raw")
        if not path:
            return value, node
        physical, target = _resolve_path(node, path, stage)
        return f"{variable}.{physical}", target

    variable_name, _, path = value[1:].partition(".")
    variable = f"${variable_name}"
    variable_node = scope.find(variable)
    if variable_node is not None:
        if not path:
            return value, variable_node
        physical, target = _resolve_path(variable_node, path, stage)
        return f"{variable}.{physical}", target

    path = value[1:]
    physical, node = _resolve_path(root, path, stage)
    return f"${physical}", node


_LOGICAL_QUERY_OPERATORS = {"$and", "$or", "$nor"}


def _convert_query(query, root, stage="$match"):
    if not isinstance(query, Mapping):
        raise InvalidQueryError("Aggregate $match must be a document")

    result = {}
    for key, value in query.items():
        if key == RAW:
            if len(query) != 1:
                raise InvalidQueryError("$raw cannot be combined with other $match keys")
            return copy.deepcopy(value)
        if key in _LOGICAL_QUERY_OPERATORS:
            if not isinstance(value, list):
                raise InvalidQueryError(f"{key} in $match must be a list")
            result[key] = [_convert_query(condition, root, stage) for condition in value]
        elif key == "$expr":
            result[key], _ = _convert_expression(value, root, stage)
        elif key in ("$jsonSchema", "$where"):
            raise InvalidQueryError(f"Use $raw for {key} in aggregate {stage!r}")
        elif isinstance(key, str) and key.startswith("$"):
            result[key] = copy.deepcopy(value)
        else:
            physical, node = _resolve_path(root, key, stage)
            result[physical] = _convert_query_value(value, node, stage)
    return result


def _convert_query_value(value, node, stage):
    if isinstance(value, Mapping):
        operator_keys = [key for key in value if isinstance(key, str) and key.startswith("$")]
        if not operator_keys:
            target_node = node.item if node is not None and node.item is not None else node
            if isinstance(value, Mapping) and target_node is not None and (
                target_node.document_type is not None or bool(target_node.children)
            ):
                return _convert_query(value, target_node, stage)
            return copy.deepcopy(value)

        result = {}
        for operator, operand in value.items():
            if operator in _LOGICAL_QUERY_OPERATORS:
                result[operator] = [
                    _convert_query_value(condition, node, stage) for condition in operand
                ]
            elif operator == "$not":
                result[operator] = _convert_query_value(operand, node, stage)
            elif operator == "$elemMatch":
                if node is not None and node.item is not None:
                    if isinstance(operand, Mapping) and any(
                        not str(key).startswith("$") for key in operand
                    ):
                        result[operator] = _convert_query(
                            operand, node.item, stage
                        )
                    else:
                        result[operator] = _convert_query_value(
                            operand, node.item, stage
                        )
                else:
                    raise InvalidQueryError(
                        f"Cannot statically resolve $elemMatch in aggregate {stage!r}; use $raw"
                    )
            elif operator == "$expr":
                result[operator], _ = _convert_expression(operand, node, stage)
            elif operator in ("$where", "$jsonSchema"):
                raise InvalidQueryError(f"Use $raw for {operator} in aggregate {stage!r}")
            elif operator == "$near" or operator.startswith("$near"):
                result[operator] = copy.deepcopy(operand)
            else:
                result[operator] = copy.deepcopy(operand)
        return result
    if isinstance(value, list):
        return copy.deepcopy(value)
    return copy.deepcopy(value)


class _PipelineTransformer:
    def __init__(self, document_type):
        self.root = _model_node(document_type)

    def transform(self, pipeline):
        converted_pipeline = []
        for index, stage in enumerate(pipeline):
            if not isinstance(stage, Mapping):
                raise TypeError(f"Aggregation pipeline stage #{index} must be a document")
            if len(stage) != 1:
                raise InvalidQueryError(f"Aggregation pipeline stage #{index} must contain exactly one operator")
            operator, spec = next(iter(stage.items()))
            converted = self.stage(operator, spec, index)
            converted_pipeline.append(converted)
        encode({"pipeline": converted_pipeline})
        return converted_pipeline

    def stage(self, operator, spec, index):
        if operator == RAW:
            if not isinstance(spec, Mapping):
                raise InvalidQueryError("$raw must contain an aggregation stage document")
            self.root = _opaque()
            return copy.deepcopy(spec)

        handler = getattr(self, f"stage_{operator.lstrip('$')}", None)
        if handler is None:
            raise InvalidQueryError(
                f"Use $raw for aggregate stage {operator!r} (#{index}); field names cannot be statically resolved"
            )
        return handler(spec, index)

    def stage_match(self, spec, index):
        return {"$match": _convert_query(spec, self.root, "$match")}

    def stage_sort(self, spec, index):
        if not isinstance(spec, Mapping):
            raise InvalidQueryError("$sort must be a document")
        result = {}
        for path, direction in spec.items():
            physical, _ = _resolve_path(self.root, path, "$sort")
            result[physical] = copy.deepcopy(direction)
        return {"$sort": result}

    def stage_limit(self, spec, index):
        return {"$limit": spec}

    def stage_skip(self, spec, index):
        return {"$skip": spec}

    def stage_unset(self, spec, index):
        paths = spec if isinstance(spec, list) else [spec]
        converted_paths = []
        for path in paths:
            physical, _ = _resolve_path(self.root, path, "$unset")
            converted_paths.append(physical)
            _remove_path(self.root, path)
        return {"$unset": converted_paths if isinstance(spec, list) else converted_paths[0]}

    def stage_project(self, spec, index):
        if not isinstance(spec, Mapping):
            raise InvalidQueryError("$project must be a document")

        items = []
        for path, value in spec.items():
            logical = "_id" if path in ("id", "pk") else path
            items.append((logical, value))

        inclusion = any(_projection_leaf(value) is True for _, value in items)
        if inclusion:
            new_root = _Node(children={})
            computed = []
            result_spec = {}
            id_explicitly_removed = False
            for path, value in items:
                leaf = _projection_leaf(value)
                if leaf is False:
                    physical, _ = _resolve_path(self.root, path, "$project")
                    result_spec[physical] = False
                    if path == "_id":
                        id_explicitly_removed = True
                    continue
                if leaf is True:
                    for included_path, _ in _projection_leaves(value, path):
                        _copy_included_path(self.root, new_root, included_path)
                        physical, _ = _resolve_path(
                            new_root, included_path, "$project"
                        )
                        result_spec[physical] = True
                else:
                    computed.append((path, value))

            for path, value in items:
                if _projection_leaf(value) is False:
                    for excluded_path, _ in _projection_leaves(value, path):
                        physical, _ = _resolve_path(self.root, excluded_path, "$project")
                        result_spec[physical] = False
                        _remove_path(new_root, excluded_path)

            if not id_explicitly_removed and "_id" not in new_root.children:
                id_node = self.root.children.get("id") or self.root.children.get("_id")
                if id_node is not None:
                    new_root.children["_id"] = _clone(id_node)

            for path, value in computed:
                converted, node = _convert_expression(value, self.root, "$project")
                _assign_computed(
                    new_root,
                    path,
                    node,
                    converted,
                    in_place=False,
                )
                result_spec[_output_physical_path(new_root, path)] = converted
            self.root = new_root
            return {"$project": result_spec}

        exclusion = any(_projection_leaf(value) is False for _, value in items)
        new_root = _Node(children={}) if exclusion else _clone(self.root)
        converted_spec = {}
        for path, value in items:
            if _projection_leaf(value) is False:
                for excluded_path, _ in _projection_leaves(value, path):
                    resolve_path = "id" if excluded_path == "_id" else excluded_path
                    physical, _ = _resolve_path(self.root, resolve_path, "$project")
                    converted_spec[physical] = False
            else:
                converted, node = _convert_expression(value, self.root, "$project")
                _assign_computed(
                    new_root,
                    path,
                    node,
                    converted,
                    in_place=False,
                )
                converted_spec[_output_physical_path(new_root, path)] = converted
        self.root = new_root
        return {"$project": converted_spec}

    def stage_set(self, spec, index):
        return self._add_fields(spec, "$set")

    def stage_addFields(self, spec, index):
        return self._add_fields(spec, "$addFields")

    def _add_fields(self, spec, operator):
        if not isinstance(spec, Mapping):
            raise InvalidQueryError(f"{operator} must be a document")
        new_root = _clone(self.root)
        result = {}
        for path, expression in spec.items():
            converted, node = _convert_expression(expression, self.root, operator)
            _assign_computed(
                new_root,
                path,
                node,
                converted,
                in_place=True,
            )
            result[_output_physical_path(new_root, path)] = converted
        self.root = new_root
        return {operator: result}

    def stage_unwind(self, spec, index):
        if isinstance(spec, str):
            path = spec[1:] if spec.startswith("$") else spec
            physical, node = _resolve_path(self.root, path, "$unwind")
            _replace_with_item(self.root, path)
            return {"$unwind": f"${physical}"}

        if not isinstance(spec, Mapping) or "path" not in spec:
            raise InvalidQueryError("Invalid $unwind stage")
        path = spec["path"][1:] if str(spec["path"]).startswith("$") else spec["path"]
        physical, node = _resolve_path(self.root, path, "$unwind")
        _replace_with_item(self.root, path)
        result = copy.deepcopy(spec)
        result["path"] = f"${physical}"
        return {"$unwind": result}

    def stage_group(self, spec, index):
        if not isinstance(spec, Mapping) or "_id" not in spec:
            raise InvalidQueryError("$group must contain an _id expression")
        result = {}
        for key, expression in spec.items():
            converted, _ = _convert_expression(expression, self.root, "$group")
            result[key] = converted
        self.root = _Node(children={key: _opaque(db_name=key) for key in result})
        return {"$group": result}

    def stage_count(self, spec, index):
        self.root = _Node(children={spec: _opaque(db_name=spec)})
        return {"$count": spec}

    def stage_rename(self, spec, index):
        if not isinstance(spec, Mapping):
            raise InvalidQueryError("$rename must be a document")
        result = {}
        for source, target in spec.items():
            source_physical, node = _resolve_path(self.root, source, "$rename")
            _remove_path(self.root, source)
            _assign_physical_node(self.root, target, _synthetic(node))
            result[source_physical] = _output_physical_path(self.root, target)
        return {"$rename": result}

    def stage_replaceRoot(self, spec, index):
        if not isinstance(spec, Mapping) or "newRoot" not in spec:
            raise InvalidQueryError("$replaceRoot requires newRoot")
        converted, node = _convert_expression(spec["newRoot"], self.root, "$replaceRoot")
        self.root = node if node is not None and not node.opaque else _opaque()
        return {"$replaceRoot": {"newRoot": converted}}

    def stage_replaceWith(self, spec, index):
        expression = spec["value"] if isinstance(spec, Mapping) and "value" in spec else spec
        converted, node = _convert_expression(expression, self.root, "$replaceWith")
        self.root = node if node is not None and not node.opaque else _opaque()
        return {"$replaceWith": converted if not isinstance(spec, Mapping) else {"value": converted}}

    def stage_facet(self, spec, index):
        if not isinstance(spec, Mapping):
            raise InvalidQueryError("$facet must be a document")
        result = {}
        for name, pipeline in spec.items():
            transformer = _PipelineTransformer.__new__(_PipelineTransformer)
            transformer.root = _clone(self.root)
            result[name] = transformer.transform(pipeline)
        self.root = _Node(children={name: _Node(item=_opaque(), db_name=name) for name in result})
        return {"$facet": result}


def _is_inclusion(value):
    return _projection_leaf(value) is True


def _projection_leaf(value):
    if value is True:
        return True
    if value is False:
        return False
    if value == 1 and isinstance(value, int):
        return True
    if value == 0 and isinstance(value, int):
        return False
    if isinstance(value, Mapping):
        leaves = [_projection_leaf(item) for item in value.values()]
        if leaves and all(item is True for item in leaves):
            return True
        if leaves and all(item is False for item in leaves):
            return False
    return None


def _projection_leaves(value, prefix):
    leaf = _projection_leaf(value)
    if leaf is not None:
        yield prefix, leaf
        return
    for key, item in value.items():
        yield from _projection_leaves(item, f"{prefix}.{key}")


def _model_child(current, part):
    if current is None or current.opaque:
        return None, part
    child = current.children.get(part)
    if child is not None:
        return child, child.db_name or part
    if current.item is not None:
        if part.isdigit():
            return current.item, part
        return current.item, part
    if current.document_type is None:
        return child, child.db_name if child is not None and child.db_name else part

    field = current.document_type._fields.get(part)
    if field is None and part in ("id", "pk"):
        part = current.document_type._meta["id_field"]
        field = current.document_type._fields.get(part)
    if field is None:
        return None, part
    return current.children.get(part, _node_for_field(field)), field.db_field


def _empty_like(node, db_name):
    return _Node(
        document_type=node.document_type,
        db_name=db_name,
        dynamic=node.dynamic,
        dict_field=node.dict_field,
        item=_Node(children={}) if node.item is not None else None,
        children={},
    )


def _copy_included_path(source_root, target_root, path):
    parts = str(path).split(".")
    source = source_root
    target = target_root
    for index, part in enumerate(parts):
        child, physical = _model_child(source, part)
        if child is None:
            raise _path_error(path, "$project")
        last = index == len(parts) - 1

        if source.item is not None and not part.isdigit():
            source = source.item
            target = target.item

        if last:
            copied = _clone(child)
            copied.db_name = physical
            target.children[part] = copied
            return copied

        existing = target.children.get(part)
        if existing is None:
            existing = _empty_like(child, physical)
            target.children[part] = existing
        target = existing
        source = child
        if source.item is not None and not part.isdigit():
            source = source.item
            if target.item is None:
                target.item = _Node(children={})
            target = target.item


def _synthetic_child(parent, part, physical):
    if parent is not None and parent.document_type is not None:
        field = parent.document_type._fields.get(part)
        if field is not None:
            return parent.children.get(part, _node_for_field(field)), field.db_field
    child = parent.children.get(part) if parent is not None and not parent.opaque else None
    if child is not None:
        return child, child.db_name or physical
    return _Node(db_name=physical, children={}), physical


def _assign_computed(root, logical_path, node, expression, in_place=False):
    parts = str(logical_path).split(".")
    current = root
    for index, part in enumerate(parts[:-1]):
        default_name = _physical_name(current, part) if current is not None and current.document_type else part
        child, physical = _synthetic_child(current, part, default_name)
        if child.opaque:
            child = _Node(db_name=physical, children={})
        elif not child.children and child.item is None and not child.document_type:
            child.db_name = physical
        if current is not None and current.document_type is not None:
            current.children[part] = child
        elif current is not None and not current.opaque:
            current.children.setdefault(part, child)
        current = child

    part = parts[-1]
    physical = (
        _physical_name(current, part)
        if in_place and current is not None and current.document_type
        else part
    )
    existing = current.children.get(part) if current is not None and not current.opaque else None
    if in_place and existing is not None and existing.document_type is not None:
        physical = existing.db_name or physical

    if node is not None and not node.opaque:
        assigned = _synthetic(node)
    else:
        assigned = _opaque(db_name=physical)
    assigned.db_name = physical
    if current is not None and not current.opaque:
        current.children[part] = assigned


def _output_physical_path(root, logical_path):
    parts = str(logical_path).split(".")
    current = root
    output = []
    for part in parts:
        child = current.children.get(part) if current is not None and not current.opaque else None
        if child is not None:
            output.append(child.db_name or part)
            current = child
            continue
        if current is not None and current.document_type is not None:
            field = current.document_type._fields.get(part)
            if field is None and part in ("id", "pk"):
                part = current.document_type._meta["id_field"]
                field = current.document_type._fields.get(part)
            if field is not None:
                output.append(field.db_field)
                current = current.children.get(part)
                continue
        output.append(child.db_name if child is not None and child.db_name else part)
        current = child
    return ".".join(output)


def _assign_physical_node(root, logical_path, node):
    parts = str(logical_path).split(".")
    current = root
    for index, part in enumerate(parts):
        physical = _physical_name(current, part) if current is not None and current.document_type else part
        last = index == len(parts) - 1
        if last:
            node.db_name = physical
            current.children[part] = node
            return
        child, _ = _synthetic_child(current, part, physical)
        current.children[part] = child
        current = child


def _replace_with_item(root, path):
    parts = str(path).split(".")
    current = root
    parent = None
    parent_part = None
    for part in parts:
        child, physical = _model_child(current, part)
        if child is None:
            raise _path_error(path, "$unwind")
        if current.item is not None and not part.isdigit():
            current = current.item
            continue
        parent, parent_part = current, part
        current = child
    if current.item is None:
        raise InvalidQueryError(f"$unwind path {path!r} is not a list field")
    unwound = _clone(current.item)
    unwound.db_name = current.db_name
    parent.children[parent_part] = unwound


def transform_pipeline(document_type, pipeline):
    if not isinstance(pipeline, (list, tuple)):
        raise TypeError("Aggregation pipeline must be a list or tuple")
    return _PipelineTransformer(document_type).transform(pipeline)
