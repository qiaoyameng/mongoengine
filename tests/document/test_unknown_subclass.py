"""Tests for reading data written by a newer deployment (unknown subclasses).

During a rolling deployment an older service can read documents whose ``_cls``
marker points to a subclass that is not registered in this process. These tests
cover both the historical behaviour (raise ``NotRegistered``) and the optional
``read_only`` unknown-subclass policy, which returns a read-only placeholder that
preserves ``_cls``, the primary key and every extra field while still allowing
the known (shared) fields to be read.
"""
import pytest

from mongoengine import (
    BooleanField,
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    EmbeddedDocumentListField,
    GenericEmbeddedDocumentField,
    GenericReferenceField,
    IntField,
    ObjectIdField,
    ReferenceField,
    StringField,
)
from mongoengine.base.common import _DocumentRegistry
from mongoengine.errors import (
    NotRegistered,
    ReadOnlyDocumentError,
    UnknownSubclassError,
)
from tests.utils import MongoDBTestCase

READ_ONLY = "read_only"


class TestUnknownSubclassPolicy(MongoDBTestCase):
    def setUp(self):
        super().setUp()

        class Part(EmbeddedDocument):
            label = StringField()
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": READ_ONLY,
            }

        class Machine(Document):
            name = StringField()
            part = EmbeddedDocumentField(Part)
            parts = EmbeddedDocumentListField(Part)
            any_part = GenericEmbeddedDocumentField()
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": READ_ONLY,
            }

        class StrictMachine(Document):
            name = StringField()
            meta = {"allow_inheritance": True}

        self.Part = Part
        self.Machine = Machine
        self.StrictMachine = StrictMachine

    def _nested_son(self, machine_id=None, part_id=None):
        # A document containing TWO as-yet-unregistered nested subclasses:
        # `Part.Bearing.SealedBearing` (two levels deep) and
        # `Part.Gasket` (unknown sibling), plus an unknown generic embedded doc.
        return {
            "_id": machine_id or ObjectIdField().to_python(None) or None,
            "_cls": "Machine",
            "name": "lathe",
            "part": {
                "_cls": "Part.Bearing.SealedBearing",
                "label": "main",
                "diameter": 12,
                "sealed": True,
            },
            "parts": [
                {"_cls": "Part.Gasket", "label": "head", "thickness": 3},
            ],
            "any_part": {
                "_cls": "Part.Valve",
                "label": "inlet",
                "pressure": 9,
            },
        }

    # ------------------------------------------------------------------
    # default policy: unchanged behaviour
    # ------------------------------------------------------------------
    def test_default_policy_still_raises_not_registered(self):
        son = {"_cls": "StrictMachine.NewModel", "name": "x", "future": 1}
        with pytest.raises(NotRegistered, match="StrictMachine.NewModel"):
            self.StrictMachine._from_son(son)

    def test_invalid_policy_value_raises(self):
        class Bad(Document):
            name = StringField()
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": "bogus",
            }

        with pytest.raises(UnknownSubclassError, match="bogus"):
            Bad._from_son({"_cls": "Bad.New", "name": "x"})

    # ------------------------------------------------------------------
    # content preservation
    # ------------------------------------------------------------------
    def test_nested_unknown_subclasses_become_placeholders(self):
        from bson import ObjectId

        machine_id = ObjectId()
        doc = self.Machine._from_son(self._nested_son(machine_id))

        assert isinstance(doc, self.Machine)
        assert doc.pk == machine_id

        bearing = doc.part
        assert bearing._is_unknown_subclass_placeholder is True
        assert bearing._class_name == "Part.Bearing.SealedBearing"
        # shared field is readable
        assert bearing.label == "main"
        # fields belonging to the unknown subclass(es) are preserved verbatim
        assert bearing._unknown_fields == {"diameter": 12, "sealed": True}
        assert bearing._unknown_raw_son == {
            "_cls": "Part.Bearing.SealedBearing",
            "label": "main",
            "diameter": 12,
            "sealed": True,
        }

        gasket = doc.parts[0]
        assert gasket._is_unknown_subclass_placeholder is True
        assert gasket._class_name == "Part.Gasket"
        assert gasket.label == "head"
        assert gasket._unknown_fields == {"thickness": 3}

        valve = doc.any_part
        assert valve._class_name == "Part.Valve"
        assert valve._unknown_fields == {"pressure": 9}

    def test_unknown_fields_are_not_silently_dropped_or_defaulted(self):
        # The placeholder must be an instance of the closest known parent but
        # never a plain parent instance whose schema drops new fields.
        doc = self.Machine._from_son(self._nested_son())
        bearing = doc.part
        assert isinstance(bearing, self.Part)
        # the new fields survive untouched (not coerced, not dropped)
        assert bearing._unknown_raw_son["diameter"] == 12
        assert "diameter" not in self.Part._fields
        # raw fields belonging to the unknown subclass are readable directly
        assert bearing.diameter == 12
        assert bearing.sealed is True
        assert doc.parts[0].thickness == 3

    def test_round_trip_through_database_preserves_everything(self):
        from bson import ObjectId

        machine_id = ObjectId()
        son = self._nested_son(machine_id)
        collection = self.Machine._get_collection()
        collection.insert_one(son)

        doc = self.Machine.objects.get(id=machine_id)
        # the parent document itself is known; saving it must rewrite the
        # unknown nested data byte-for-byte
        doc.name = "lathe-2"
        doc.save()

        raw_after = collection.find_one({"_id": machine_id})
        assert raw_after["part"] == son["part"]
        assert raw_after["parts"] == son["parts"]
        assert raw_after["any_part"] == son["any_part"]
        assert raw_after["name"] == "lathe-2"

    # ------------------------------------------------------------------
    # write permission boundary (all write entry points rejected)
    # ------------------------------------------------------------------
    def test_top_level_placeholder_write_methods_rejected(self):
        class Vehicle(Document):
            name = StringField()
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": READ_ONLY,
            }

        for method_name in ("save", "delete", "update", "modify", "reload"):
            with self.subTest(method_name=method_name):
                doc = Vehicle._from_son(
                    {"_cls": "Vehicle.Spaceship", "name": "x", "warp": 5}
                )
                assert doc._is_unknown_subclass_placeholder is True
                with pytest.raises(
                    ReadOnlyDocumentError, match="Vehicle.Spaceship"
                ):
                    getattr(doc, method_name)()

    def test_embedded_placeholder_save_and_delete_rejected(self):
        doc = self.Machine._from_son(self._nested_son())
        bearing = doc.part
        with pytest.raises(ReadOnlyDocumentError):
            bearing.save()
        with pytest.raises(ReadOnlyDocumentError):
            bearing.delete()

    def test_declared_field_mutation_rejected(self):
        doc = self.Machine._from_son(self._nested_son())
        bearing = doc.part
        with pytest.raises(ReadOnlyDocumentError):
            bearing.label = "changed"
        with pytest.raises(ReadOnlyDocumentError):
            bearing["label"] = "changed"
        with pytest.raises(ReadOnlyDocumentError):
            del bearing.label
        # value is untouched
        assert bearing.label == "main"

    def test_top_level_placeholder_field_mutation_rejected(self):
        class Vehicle(Document):
            name = StringField()
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": READ_ONLY,
            }

        doc = Vehicle._from_son(
            {"_cls": "Vehicle.Spaceship", "name": "x", "warp": 5}
        )
        with pytest.raises(ReadOnlyDocumentError):
            doc.name = "y"
        with pytest.raises(ReadOnlyDocumentError):
            doc["name"] = "y"
        assert doc.name == "x"

    # ------------------------------------------------------------------
    # reference decoding keeps its strict boundary
    # ------------------------------------------------------------------
    def test_unknown_generic_reference_is_not_silently_instantiated(self):
        from bson import DBRef, ObjectId

        class Animal(Document):
            name = StringField()
            meta = {
                "allow_inheritance": True,
                "collection": "unk_generic_ref_animal",
            }

        class Owner(Document):
            pet = GenericReferenceField()
            meta = {
                "allow_inheritance": True,
                "collection": "unk_generic_ref_owner",
                "unknown_subclass_policy": READ_ONLY,
            }

        animal_id = ObjectId()
        Animal._get_collection().insert_one(
            {"_id": animal_id, "_cls": "Animal.Griffin", "name": "g", "claws": 3}
        )
        owner = Owner(pet={"_cls": "Animal.Griffin", "_ref": DBRef("animal", animal_id)})
        owner.save()
        reloaded = Owner.objects.get(id=owner.id)
        # even with read_only policy, dereferencing an unknown reference keeps
        # raising rather than returning a writable parent instance
        with pytest.raises(NotRegistered, match="Animal.Griffin"):
            _ = reloaded.pet

    def test_unknown_reference_field_keeps_raising(self):
        from bson import ObjectId

        class Animal(Document):
            name = StringField()
            meta = {
                "allow_inheritance": True,
                "collection": "unk_ref_animal",
            }

        from bson import DBRef

        class Owner(Document):
            pet = ReferenceField(Animal, dbref=True)
            meta = {
                "allow_inheritance": True,
                "collection": "unk_ref_owner",
                "unknown_subclass_policy": READ_ONLY,
            }

        animal_id = ObjectId()
        Animal._get_collection().insert_one(
            {"_id": animal_id, "_cls": "Animal.Griffin", "name": "g"}
        )
        Owner._get_collection().insert_one(
            {
                "_id": ObjectId(),
                "_cls": "Owner",
                "pet": DBRef("animal", animal_id, cls="Animal.Griffin"),
            }
        )
        owner = Owner.objects.first()
        with pytest.raises(NotRegistered, match="Animal.Griffin"):
            _ = owner.pet

    # ------------------------------------------------------------------
    # recovery once the subclass is registered
    # ------------------------------------------------------------------
    def test_decodes_to_normal_model_after_registration(self):
        from bson import ObjectId

        machine_id = ObjectId()
        son = self._nested_son(machine_id)

        # before registration -> placeholder
        before = self.Machine._from_son(son)
        assert before.part._is_unknown_subclass_placeholder is True

        # register the missing subclasses (as an import on a new deploy would)
        class Bearing(self.Part):
            diameter = IntField()

        class SealedBearing(Bearing):
            sealed = BooleanField()

        class Gasket(self.Part):
            thickness = IntField()

        class Valve(self.Part):
            pressure = IntField()

        try:
            after = self.Machine._from_son(son)
            assert type(after.part) is SealedBearing
            assert after.part.diameter == 12
            assert after.part.sealed is True
            assert type(after.parts[0]) is Gasket
            assert after.parts[0].thickness == 3
            assert type(after.any_part) is Valve
            assert after.any_part.pressure == 9
            # normal model is writable again
            after.part.label = "updated"
            assert after.part.label == "updated"
        finally:
            for cls in (SealedBearing, Gasket, Valve, Bearing):
                _DocumentRegistry.unregister(cls._class_name)

    def test_queryset_returns_unknown_subclasses_as_placeholders(self):
        from bson import ObjectId

        class UnkQueryVehicle(Document):
            name = StringField()
            meta = {
                "allow_inheritance": True,
                "collection": "unk_queryset_vehicle",
                "unknown_subclass_policy": READ_ONLY,
            }

        collection = UnkQueryVehicle._get_collection()
        collection.insert_many(
            [
                {"_id": ObjectId(), "_cls": "UnkQueryVehicle", "name": "plain"},
                {
                    "_id": ObjectId(),
                    "_cls": "UnkQueryVehicle.Spaceship",
                    "name": "falcon",
                    "warp": 9,
                },
                {
                    "_id": ObjectId(),
                    "_cls": "UnkQueryVehicle.Spaceship.DeepSpace",
                    "name": "voyager",
                },
                # unrelated document must still be filtered out
                {"_id": ObjectId(), "_cls": "SomethingElse", "name": "x"},
            ]
        )

        names_and_flags = [
            (d._class_name, d._is_unknown_subclass_placeholder)
            for d in UnkQueryVehicle.objects.order_by("name")
        ]
        assert names_and_flags == [
            ("UnkQueryVehicle.Spaceship", True),
            ("UnkQueryVehicle", False),
            ("UnkQueryVehicle.Spaceship.DeepSpace", True),
        ]
        spaceship = UnkQueryVehicle.objects(
            _cls="UnkQueryVehicle.Spaceship"
        ).first()
        assert spaceship.warp == 9

    def test_default_queryset_does_not_return_unknown_subclasses(self):
        from bson import ObjectId

        collection = self.StrictMachine._get_collection()
        collection.insert_many(
            [
                {"_id": ObjectId(), "_cls": "StrictMachine", "name": "ok"},
                {
                    "_id": ObjectId(),
                    "_cls": "StrictMachine.NewModel",
                    "name": "future",
                },
            ]
        )
        # historical behaviour: unknown descendants are not matched by the
        # strict class' _cls query
        assert self.StrictMachine.objects.count() == 1
        assert self.StrictMachine.objects.first().name == "ok"

    def test_registry_is_not_polluted_by_placeholder_classes(self):
        self.Machine._from_son(self._nested_son())
        assert _DocumentRegistry.get_or_none("Part.Gasket") is None
        assert _DocumentRegistry.get_or_none("Part.Bearing.SealedBearing") is None
