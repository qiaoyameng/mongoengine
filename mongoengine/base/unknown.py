from bson import SON

from mongoengine.base.common import _DocumentRegistry
from mongoengine.errors import OperationError

__all__ = ("PRESERVE_UNKNOWN_SUBCLASS_POLICY", "UnknownSubclassCarrier")

# Value of the ``unknown_subclass_policy`` meta option that turns unknown
# ``_cls`` decoding into a read-only carrier instead of raising NotRegistered.
PRESERVE_UNKNOWN_SUBCLASS_POLICY = "preserve"


class UnknownSubclassCarrier:
    """Read-only carrier for a document whose ``_cls`` is not registered.

    During a rolling deployment, an old service may read documents written
    by a newer service whose subclass has not been imported (registered) in
    the old process yet. When the queried class opts in with
    ``meta = {"unknown_subclass_policy": "preserve"}``, decoding such a
    document returns this carrier instead of raising
    :class:`~mongoengine.errors.NotRegistered`.

    The carrier:

    * preserves the original BSON exactly (``_cls``, every raw field and the
      primary key), so nothing written by the newer service is lost;
    * exposes the fields it shares with the queried class as read-only
      attributes;
    * refuses every write entry point (``save``, ``delete``, ``update``,
      ``modify``, ``reload``) and any attribute mutation.

    It never instantiates the parent class, so new fields are never
    overwritten with defaults. Once the real class is registered, call
    :meth:`restore` (or simply re-decode the BSON) to get a normal model
    instance.
    """

    __slots__ = ("_parent_cls", "_cls_name", "_raw_data", "_common_data")

    def __init__(self, parent_cls, cls_name, son):
        object.__setattr__(self, "_parent_cls", parent_cls)
        object.__setattr__(self, "_cls_name", cls_name)
        object.__setattr__(self, "_raw_data", SON(son))

        reverse_db_field_map = getattr(parent_cls, "_reverse_db_field_map", {})
        common = {}
        for key, value in son.items():
            field_name = reverse_db_field_map.get(str(key), str(key))
            field = parent_cls._fields.get(field_name)
            if field is None:
                continue
            if value is None:
                common[field_name] = None
                continue
            try:
                common[field_name] = field.to_python(value)
            except Exception:
                # A shared field that fails to decode must not lose its raw value
                common[field_name] = value
        object.__setattr__(self, "_common_data", common)

    @property
    def _cls(self):
        """Name of the unregistered class, as stored in the database."""
        return self._cls_name

    @property
    def pk(self):
        """Primary key preserved from the raw BSON (None for embedded docs)."""
        return self._common_data.get("id", self._raw_data.get("_id"))

    @property
    def id(self):
        """Alias for :attr:`pk`."""
        return self.pk

    def __getattr__(self, name):
        if name == "_common_data":
            raise AttributeError(name)
        common = object.__getattribute__(self, "_common_data")
        try:
            return common[name]
        except KeyError:
            raise AttributeError(
                "%s (unregistered class %r) only exposes the fields shared "
                "with %s; %r is not one of them. The raw value is preserved "
                "in to_mongo()."
                % (
                    type(self).__name__,
                    self._cls_name,
                    self._parent_cls._class_name,
                    name,
                )
            )

    def __setattr__(self, name, value):
        raise OperationError(
            "%s is read-only: cannot set %r on unregistered class %r. "
            "Register the class and call restore() before mutating it."
            % (type(self).__name__, name, self._cls_name)
        )

    def __delattr__(self, name):
        raise OperationError(
            "%s is read-only: cannot delete %r on unregistered class %r."
            % (type(self).__name__, name, self._cls_name)
        )

    def _deny_write(self, operation):
        raise OperationError(
            "Cannot %s unregistered class %r: this process only holds a "
            "read-only carrier of the raw BSON. Register the class and call "
            "restore() (or re-decode the BSON) before writing."
            % (operation, self._cls_name)
        )

    def save(self, *args, **kwargs):
        self._deny_write("save")

    def delete(self, *args, **kwargs):
        self._deny_write("delete")

    def update(self, *args, **kwargs):
        self._deny_write("update")

    def modify(self, *args, **kwargs):
        self._deny_write("modify")

    def reload(self, *args, **kwargs):
        self._deny_write("reload")

    def to_mongo(self, *args, **kwargs):
        """Return the preserved BSON, exactly as it was read."""
        return SON(self._raw_data)

    def restore(self):
        """Re-decode the preserved BSON into the real model class.

        Raises :class:`~mongoengine.errors.NotRegistered` if the class is
        still not registered.
        """
        doc_cls = _DocumentRegistry.get(self._cls_name)
        return doc_cls._from_son(self._raw_data)

    def __repr__(self):
        return "<%s: %r (unregistered), pk=%r>" % (
            type(self).__name__,
            self._cls_name,
            self.pk,
        )
