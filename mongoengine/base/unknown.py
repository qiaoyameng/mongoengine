"""Support for reading documents whose ``_cls`` marker is not registered.

During a rolling deployment an older version of an application can read data
that was written by a newer version which defines additional document
subclasses.  In that situation the class named by ``_cls`` is not present in
the :class:`~mongoengine.base.common._DocumentRegistry`.

By default decoding such data raises
:class:`~mongoengine.errors.NotRegistered`, which preserves the historical
behaviour.  When the ``unknown_subclass_policy`` meta option is set to
``"read_only"`` the data is decoded into a *read-only placeholder* instance of
the closest known parent class.  The placeholder keeps:

* the original ``_cls`` value,
* the primary key,
* every field declared on the known parent classes (decoded normally), and
* all extra fields written by the unknown subclass, stored untouched.

The placeholder cannot be saved, deleted or mutated, which guarantees that the
fields owned by the unknown subclass can never be silently overwritten by the
parent class defaults.  Once the subclass is imported (and therefore
registered), decoding the same data again returns a normal instance of that
subclass.
"""
import copy

from mongoengine.errors import (
    InvalidDocumentError,
    NotRegistered,
    ReadOnlyDocumentError,
    UnknownSubclassError,
)

__all__ = (
    "UNKNOWN_SUBCLASS_ERROR",
    "UNKNOWN_SUBCLASS_READ_ONLY",
    "UNKNOWN_SUBCLASS_POLICIES",
    "UnknownSubclassDocumentMixin",
    "build_unknown_subclass_document",
    "decode_unknown_aware",
    "resolve_subclass",
)


UNKNOWN_SUBCLASS_ERROR = "error"
UNKNOWN_SUBCLASS_READ_ONLY = "read_only"

UNKNOWN_SUBCLASS_POLICIES = frozenset(
    {UNKNOWN_SUBCLASS_ERROR, UNKNOWN_SUBCLASS_READ_ONLY}
)

_READ_ONLY_MESSAGE = (
    "This document is a read-only placeholder for the unregistered subclass "
    "`{cls}`. Import/register the matching Document subclass before mutating, "
    "saving or deleting it."
)


class UnknownSubclassDocumentMixin:
    """Mixin flagging instances holding data of an unregistered subclass.

    The concrete placeholder classes are created dynamically by
    :func:`build_unknown_subclass_document`.  They subclass the closest known
    document class so that all shared fields remain readable, but every write
    entry point is rejected.
    """

    _is_unknown_subclass_placeholder = True

    #: The ``_cls`` value found in the source data.
    _unknown_subclass_name = None

    #: The untouched extra fields owned by the unknown subclass.
    _unknown_fields = None

    @classmethod
    def _is_unknown_subclass(cls):
        return True

    def _read_only_error(self):
        return ReadOnlyDocumentError(
            _READ_ONLY_MESSAGE.format(cls=self._unknown_subclass_name)
        )

    def save(self, *args, **kwargs):
        raise self._read_only_error()

    def delete(self, *args, **kwargs):
        raise self._read_only_error()

    def update(self, *args, **kwargs):
        raise self._read_only_error()

    def modify(self, *args, **kwargs):
        raise self._read_only_error()

    def reload(self, *args, **kwargs):
        raise self._read_only_error()

    def __setattr__(self, name, value):
        # Internal bookkeeping (everything starting with "_") is required to
        # build the placeholder itself; user-facing, non-field attributes are
        # rejected once the placeholder has been fully initialised. Declared
        # fields are data descriptors and are rejected through
        # _mark_as_changed instead.
        if (
            not name.startswith("_")
            and getattr(self, "_initialised", False)
            and name not in type(self)._fields
        ):
            raise self._read_only_error()
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if not name.startswith("_") and getattr(self, "_initialised", False):
            raise self._read_only_error()
        super().__delattr__(name)

    def __getattr__(self, name):
        # Only called when normal attribute lookup fails: expose the raw
        # values of fields owned by the unknown subclass in read-only fashion.
        if name.startswith("_"):
            raise AttributeError(name)
        unknown = self.__dict__.get("_unknown_fields")
        if unknown is not None and name in unknown:
            return self._data[name]
        raise AttributeError(name)

    def __setitem__(self, name, value):
        raise self._read_only_error()

    def _mark_as_changed(self, key):
        # Every write to a declared field funnels through here (via the field
        # descriptors); reject it on a fully built placeholder.
        if getattr(self, "_initialised", False):
            raise self._read_only_error()

    def _mark_as_unset(self, key):
        if getattr(self, "_initialised", False):
            raise self._read_only_error()

    def to_mongo(self, *args, **kwargs):
        # Re-emit the exact source document so the unknown fields are
        # preserved when the parent document (in the embedded case) is saved.
        return copy.deepcopy(self._unknown_raw_son)

    def __getstate__(self):
        return {
            "_unknown_subclass_name": self._unknown_subclass_name,
            "_unknown_raw_son": self._unknown_raw_son,
        }

    def __setstate__(self, state):
        # Rebuild through the public builder so a placeholder stays a
        # placeholder (and becomes a normal instance if the subclass was
        # registered in the meantime, e.g. after an import). BaseDocument
        # stores its state in slots, so those have to be copied explicitly in
        # addition to the regular instance __dict__.
        from mongoengine.base.common import _DocumentRegistry

        class_name = state["_unknown_subclass_name"]
        raw_son = state["_unknown_raw_son"]
        registered_cls = _DocumentRegistry.get_or_none(class_name)
        if registered_cls is not None:
            rebuilt = registered_cls._from_son(raw_son)
        else:
            base_cls = _find_known_ancestor(class_name)
            rebuilt = build_unknown_subclass_document(
                base_cls, raw_son, class_name
            )
        for slot in (
            "_changed_fields",
            "_unset_fields",
            "_has_change_tracking_baseline",
            "_initialised",
            "_created",
            "_data",
            "_dynamic_fields",
            "_auto_id_field",
            "_db_field_map",
        ):
            if hasattr(rebuilt, slot):
                setattr(self, slot, getattr(rebuilt, slot))
        self.__dict__.update(rebuilt.__dict__)
        if hasattr(rebuilt, "_instance"):
            try:
                self._instance = rebuilt._instance
            except Exception:
                pass


def _build_placeholder_class(base_cls, class_name):
    """Create (once) an unregistered placeholder subclass of ``base_cls``."""
    cache = getattr(base_cls, "_unknown_subclass_classes", None)
    if cache is None:
        cache = {}
        base_cls._unknown_subclass_classes = cache
    placeholder_cls = cache.get(class_name)
    if placeholder_cls is not None:
        return placeholder_cls

    from mongoengine.base.common import _DocumentRegistry

    metaclass = type(base_cls)
    placeholder_name = "_UnknownSubclass_%s_%s" % (
        base_cls.__name__,
        class_name.replace(".", "_"),
    )
    # Feed attributes as they would look after the metaclass pre-processing
    # of a normal subclass definition: no `my_metaclass` marker (which would
    # flag this as the abstract base) and a concrete, non-abstract `_meta`.
    meta = dict(base_cls._meta)
    meta["abstract"] = False
    # The placeholder shares the parent collection and never queries on its
    # own; dropping the explicit key avoids the "setting a collection on a
    # subclass" warning during the synthetic class creation.
    meta.pop("collection", None)
    attrs = {
        "__module__": base_cls.__module__,
        "_is_base_cls": False,
        "_is_document": base_cls._is_document,
        "_meta": meta,
    }
    placeholder_cls = metaclass.__new__(
        metaclass,
        placeholder_name,
        (UnknownSubclassDocumentMixin, base_cls),
        attrs,
    )
    # The metaclass registered the synthetic class under its generated name;
    # remove it so it never shadows a real subclass loaded later and never
    # appears in _subclasses based queries.
    _DocumentRegistry.unregister(placeholder_cls._class_name)
    # Override the metadata derived from the synthetic class name.
    placeholder_cls._class_name = class_name
    placeholder_cls._superclasses = base_cls._superclasses + (
        base_cls._class_name,
    )
    placeholder_cls._subclasses = (class_name,)
    placeholder_cls._types = (class_name,)
    cache[class_name] = placeholder_cls
    return placeholder_cls


def _decode_shared_fields(cls, son, _auto_dereference):
    """Apply field conversion for the fields known on ``cls``.

    This mirrors :meth:`BaseDocument._from_son` without dropping any of the
    keys that are not declared on the known (parent) class.
    """
    import copy as _copy

    data = {}
    for key, value in son.items():
        key = str(key)
        key = cls._db_field_map.get(key, key)
        data[key] = value

    fields = cls._fields
    if not _auto_dereference:
        fields = _copy.deepcopy(fields)

    errors_dict = {}
    for field_name, field in fields.items():
        field.set_auto_dereferencing(_auto_dereference)
        if field.db_field in data:
            value = data[field.db_field]
            try:
                data[field_name] = value if value is None else field.to_python(value)
                if field_name != field.db_field:
                    del data[field.db_field]
            except (AttributeError, ValueError) as exc:
                errors_dict[field_name] = exc

    if errors_dict:
        errors = "\n".join(
            [f"Field '{k}' - {v}" for k, v in errors_dict.items()]
        )
        raise InvalidDocumentError(
            "Invalid data to create a `{}` instance.\n{}".format(
                cls._class_name, errors
            )
        )
    return data, fields


def _find_known_ancestor(class_name, fallback=None):
    """Return the closest registered ancestor named by a dotted _cls path."""
    from mongoengine.base.common import _DocumentRegistry

    parts = class_name.split(".")
    for index in range(len(parts) - 1, 0, -1):
        ancestor = _DocumentRegistry.get_or_none(".".join(parts[:index]))
        if ancestor is not None:
            return ancestor
    return fallback


def build_unknown_subclass_document(base_cls, son, class_name, _auto_dereference=True):
    """Build a read-only placeholder for ``son`` belonging to ``class_name``.

    :param base_cls: the closest registered document class able to decode the
        shared fields (usually the parent referenced by the dotted ``_cls``).
    :param son: the raw SON/dict read from the database (never mutated).
    :param class_name: the unregistered ``_cls`` value.
    """
    placeholder_cls = _build_placeholder_class(base_cls, class_name)

    raw_son = copy.deepcopy(son)
    data, fields = _decode_shared_fields(placeholder_cls, raw_son, _auto_dereference)

    # ``strict=False`` lets the regular constructor accept the keys that are
    # not declared on the known parent class: they are stored verbatim in
    # ``_data`` instead of raising FieldDoesNotExist. They are never converted
    # (they belong to a class this process does not know about).
    known = set(placeholder_cls._fields) | {"id", "pk", "_cls", "_text_score"}
    shared = {k: v for k, v in data.items() if k in known}
    unknown = {k: v for k, v in data.items() if k not in known}

    original_meta = placeholder_cls._meta
    try:
        placeholder_cls._meta = dict(original_meta)
        placeholder_cls._meta["strict"] = False
        obj = placeholder_cls(__auto_convert=False, _created=False, **shared)
    finally:
        placeholder_cls._meta = original_meta

    obj._has_change_tracking_baseline = True
    if not _auto_dereference:
        obj._fields = fields

    # Unknown fields are kept verbatim and exposed read-only. They are not
    # part of _fields_ordered so the normal machinery never rewrites them;
    # serialization uses _unknown_raw_son to guarantee full round-tripping.
    for key, value in unknown.items():
        obj._data[key] = value
    obj._unknown_subclass_name = class_name
    obj._unknown_fields = unknown
    obj._unknown_raw_son = raw_son
    obj._changed_fields = []
    obj._unset_fields = []
    return obj


def resolve_subclass(cls, son, _auto_dereference=True):
    """Resolve ``son`` to the right class instance honoring the policy.

    Returns an instance (possibly a read-only placeholder), or raises
    :class:`~mongoengine.errors.NotRegistered` with the default policy.
    """
    from mongoengine.base.common import _DocumentRegistry

    class_name = son.get("_cls", cls._class_name)
    if class_name == cls._class_name:
        return None

    registered_cls = _DocumentRegistry.get_or_none(class_name)
    if registered_cls is not None:
        return registered_cls

    policy = cls._meta.get("unknown_subclass_policy", UNKNOWN_SUBCLASS_ERROR)
    if policy not in UNKNOWN_SUBCLASS_POLICIES:
        raise UnknownSubclassError(
            "Unknown `unknown_subclass_policy` %r on `%s`. Valid policies "
            "are: %s"
            % (policy, cls._class_name, sorted(UNKNOWN_SUBCLASS_POLICIES))
        )
    if policy == UNKNOWN_SUBCLASS_ERROR:
        raise NotRegistered(
            "`%s` has not been registered in the document registry. "
            "Importing the document class automatically registers it, has it "
            "been imported?" % class_name
        )

    # Find the closest registered ancestor named by the dotted _cls path.
    base_cls = _find_known_ancestor(class_name, fallback=cls)

    return build_unknown_subclass_document(
        base_cls, son, class_name, _auto_dereference=_auto_dereference
    )


def decode_unknown_aware(value, policy_cls, *, is_reference=False, dereference=None):
    """Decode a SON dict carrying a ``_cls`` marker, honoring the policy.

    Used by generic/dynamic/reference fields that resolve the target class
    themselves rather than through ``cls._from_son``.

    :param value: the SON dict (contains ``_cls``).
    :param policy_cls: the class whose ``unknown_subclass_policy`` applies.
        When ``None`` (purely generic fields with no owning class), references
        default to the historical ``error`` policy while embedded documents use
        the closest registered ancestor as fallback parent.
    :param is_reference: if True the dict encodes a generic reference
        (``{"_cls": ..., "_ref": DBRef(...)}``) which is only dereferenced when
        the target class is known; otherwise it is left untouched.
    :param dereference: zero-argument callable performing the actual DB fetch.
    """
    from mongoengine.base.common import _DocumentRegistry

    class_name = value["_cls"]
    registered_cls = _DocumentRegistry.get_or_none(class_name)
    if registered_cls is not None:
        if dereference is not None:
            value = dereference(registered_cls)
        return registered_cls._from_son(value)

    if is_reference:
        # Reference decoding always keeps its historical permission boundary:
        # an unknown target class is never silently instantiated as the parent
        # document (that would produce a writable object with an ambiguous
        # identity). The raw reference marker (``_cls`` + ``_ref``) stays
        # untouched in the owning document's data and is re-saved verbatim.
        raise NotRegistered(
            "`%s` has not been registered in the document registry. "
            "Importing the document class automatically registers it, has it "
            "been imported?" % class_name
        )

    if policy_cls is None:
        raise NotRegistered(
            "`%s` has not been registered in the document registry. "
            "Importing the document class automatically registers it, has it "
            "been imported?" % class_name
        )

    # For embedded documents, resolve against a known embedded ancestor named
    # by the dotted path. Never fall back to a top-level owning Document (a
    # different type) just because its policy opted into read_only.
    from mongoengine.document import EmbeddedDocument

    if not issubclass(policy_cls, EmbeddedDocument):
        embedded_base = _find_known_ancestor(class_name)
        if embedded_base is not None and issubclass(
            embedded_base, EmbeddedDocument
        ):
            return build_unknown_subclass_document(
                embedded_base, value, class_name
            )
        raise NotRegistered(
            "`%s` has not been registered in the document registry. "
            "Importing the document class automatically registers it, has it "
            "been imported?" % class_name
        )

    resolved = resolve_subclass(policy_cls, value)
    return resolved
