import pytest
from bson import SON, ObjectId

from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    GenericEmbeddedDocumentField,
    IntField,
    ReferenceField,
    StringField,
)
from mongoengine.base.common import _DocumentRegistry
from mongoengine.base.unknown import UnknownSubclassCarrier
from mongoengine.errors import NotRegistered, OperationError
from tests.utils import MongoDBTestCase, get_as_pymongo


class CarrierPayload(EmbeddedDocument):
    meta = {"allow_inheritance": True, "unknown_subclass_policy": "preserve"}
    name = StringField(default="anon")


class CarrierHolder(Document):
    title = StringField()
    payload = EmbeddedDocumentField(CarrierPayload)


class CarrierEvent(Document):
    meta = {"allow_inheritance": True, "unknown_subclass_policy": "preserve"}
    name = StringField()


class StrictPayload(EmbeddedDocument):
    meta = {"allow_inheritance": True}
    name = StringField()


class StrictHolder(Document):
    payload = EmbeddedDocumentField(StrictPayload)


class GenericHolder(Document):
    payload = GenericEmbeddedDocumentField()


# Note: embedded document subclasses use a flat ``_cls`` (unlike Document
# subclasses, which are dotted with their parents' names).
NEW_PAYLOAD_CLS = "ScheduledJob"
NEW_EVENT_CLS = "CarrierEvent.ScheduledEvent"


def make_scheduled_job_cls():
    class ScheduledJob(CarrierPayload):
        cron = StringField()
        retries = IntField()

    return ScheduledJob


def nested_bson():
    """BSON for a Holder written by a newer service, embedding a subclass
    this process has not registered."""
    return SON(
        [
            ("_id", ObjectId("5f8d0d55b54764421b7156c7")),
            ("title", "nightly"),
            (
                "payload",
                SON(
                    [
                        ("_cls", NEW_PAYLOAD_CLS),
                        ("name", "cleanup"),
                        ("cron", "0 3 * * *"),
                        ("retries", 3),
                    ]
                ),
            ),
        ]
    )


def top_level_bson():
    return SON(
        [
            ("_id", ObjectId("5f8d0d55b54764421b7156c8")),
            ("_cls", NEW_EVENT_CLS),
            ("name", "deploy"),
            ("zone", "eu-2"),
        ]
    )


class TestUnknownSubclassCarrierDecoding:
    def test_nested_unknown_subclass_returns_carrier_preserving_raw_bson(self):
        holder = CarrierHolder._from_son(nested_bson())

        carrier = holder.payload
        assert isinstance(carrier, UnknownSubclassCarrier)
        assert carrier._cls == NEW_PAYLOAD_CLS

        # Common fields shared with the declared type are readable
        assert carrier.name == "cleanup"

        # Fields of the unregistered subclass are not exposed as attributes...
        with pytest.raises(AttributeError):
            carrier.cron

        # ...but the raw BSON (incl. _cls and the new fields) is preserved
        raw = carrier.to_mongo()
        assert raw == nested_bson()["payload"]
        assert raw["cron"] == "0 3 * * *"
        assert raw["retries"] == 3

        # Re-serializing the parent document preserves the nested payload
        # byte-for-byte instead of dropping the new fields
        assert CarrierHolder._from_son(nested_bson()).to_mongo() == nested_bson()

    def test_top_level_unknown_subclass_preserves_cls_fields_and_pk(self):
        carrier = CarrierEvent._from_son(top_level_bson())

        assert isinstance(carrier, UnknownSubclassCarrier)
        assert carrier._cls == NEW_EVENT_CLS
        assert carrier.pk == top_level_bson()["_id"]
        assert carrier.id == top_level_bson()["_id"]
        assert carrier.name == "deploy"
        assert carrier.to_mongo() == top_level_bson()

    def test_carrier_is_not_silently_instantiated_as_parent(self):
        carrier = CarrierHolder._from_son(nested_bson()).payload

        # A parent-class instance would apply defaults and drop new fields;
        # the carrier must keep the raw document exactly as stored.
        assert not isinstance(carrier, CarrierPayload)
        assert carrier.to_mongo()["cron"] == "0 3 * * *"
        assert "retries" in carrier.to_mongo()

    def test_all_write_entry_points_are_rejected(self):
        carrier = CarrierEvent._from_son(top_level_bson())

        for operation in ("save", "delete", "update", "modify", "reload"):
            with pytest.raises(OperationError):
                getattr(carrier, operation)()

        with pytest.raises(OperationError):
            carrier.name = "changed"
        with pytest.raises(OperationError):
            del carrier.name

    def test_default_policy_still_raises_not_registered(self):
        strict_bson = SON(nested_bson())
        strict_bson["payload"] = SON(
            [("_cls", "StrictPayload.Unknown"), ("name", "x"), ("extra", 1)]
        )
        with pytest.raises(NotRegistered):
            StrictHolder._from_son(strict_bson)

        with pytest.raises(NotRegistered):
            GenericHolder._from_son(SON([("payload", {"_cls": "Anything.Unknown"})]))

        class StrictEvent(Document):
            meta = {"allow_inheritance": True}
            name = StringField()

        with pytest.raises(NotRegistered):
            StrictEvent._from_son(SON([("_cls", "StrictEvent.Unknown"), ("name", "x")]))

    def test_decoding_recovers_normal_model_once_class_is_registered(self):
        ScheduledJob = make_scheduled_job_cls()
        try:
            # Re-decoding the same BSON now yields the real model
            holder = CarrierHolder._from_son(nested_bson())
            assert isinstance(holder.payload, ScheduledJob)
            assert holder.payload.name == "cleanup"
            assert holder.payload.cron == "0 3 * * *"
            assert holder.payload.retries == 3
        finally:
            _DocumentRegistry.unregister(NEW_PAYLOAD_CLS)

    def test_carrier_restore_recovers_normal_model_once_class_is_registered(self):
        carrier = CarrierHolder._from_son(nested_bson()).payload
        with pytest.raises(NotRegistered):
            carrier.restore()

        ScheduledJob = make_scheduled_job_cls()
        try:
            restored = carrier.restore()
            assert isinstance(restored, ScheduledJob)
            assert restored.cron == "0 3 * * *"
            assert restored.retries == 3
        finally:
            _DocumentRegistry.unregister(NEW_PAYLOAD_CLS)


class TestUnknownSubclassWithDatabase(MongoDBTestCase):
    def tearDown(self):
        self.db.carrier_account.drop()
        self.db.carrier_team.drop()
        self.db.plain_account.drop()
        self.db.plain_team.drop()
        self.db.carrier_job_holder.drop()

    def test_query_and_reference_decode_to_carrier(self):
        class CarrierAccount(Document):
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": "preserve",
            }
            name = StringField()

        class CarrierTeam(Document):
            owner = ReferenceField(CarrierAccount)

        class VipCarrierAccount(CarrierAccount):
            tier = StringField()

        account = VipCarrierAccount(name="vip", tier="gold").save()
        team = CarrierTeam(owner=account).save()

        _DocumentRegistry.unregister("CarrierAccount.VipCarrierAccount")
        try:
            fetched = CarrierAccount.objects.get(id=account.id)
            assert isinstance(fetched, UnknownSubclassCarrier)
            assert fetched._cls == "CarrierAccount.VipCarrierAccount"
            assert fetched.pk == account.pk
            assert fetched.name == "vip"
            assert fetched.to_mongo()["tier"] == "gold"
            with pytest.raises(OperationError):
                fetched.save()
            with pytest.raises(OperationError):
                fetched.delete()

            # Reference decoding boundary: the declared type's policy applies
            owner = CarrierTeam.objects.get(id=team.id).owner
            assert isinstance(owner, UnknownSubclassCarrier)
            assert owner.name == "vip"
        finally:
            _DocumentRegistry.register(VipCarrierAccount)

        # Once the class is registered again, decoding recovers the model
        assert isinstance(CarrierAccount.objects.get(id=account.id), VipCarrierAccount)

    def test_reference_to_unknown_subclass_raises_by_default(self):
        class PlainAccount(Document):
            meta = {"allow_inheritance": True}
            name = StringField()

        class PlainTeam(Document):
            owner = ReferenceField(PlainAccount)

        class VipPlainAccount(PlainAccount):
            tier = StringField()

        account = VipPlainAccount(name="vip").save()
        team = PlainTeam(owner=account).save()

        _DocumentRegistry.unregister("PlainAccount.VipPlainAccount")
        try:
            with pytest.raises(NotRegistered):
                PlainAccount.objects.get(id=account.id)
            with pytest.raises(NotRegistered):
                PlainTeam.objects.get(id=team.id).owner
        finally:
            _DocumentRegistry.register(VipPlainAccount)

    def test_parent_document_roundtrip_preserves_unknown_embedded_subclass(self):
        class CarrierJob(EmbeddedDocument):
            meta = {
                "allow_inheritance": True,
                "unknown_subclass_policy": "preserve",
            }
            name = StringField()

        class ScheduledCarrierJob(CarrierJob):
            cron = StringField()

        class CarrierJobHolder(Document):
            title = StringField()
            job = EmbeddedDocumentField(CarrierJob)

        holder = CarrierJobHolder(
            title="t", job=ScheduledCarrierJob(name="n", cron="* * * * *")
        ).save()

        _DocumentRegistry.unregister("ScheduledCarrierJob")
        try:
            fetched = CarrierJobHolder.objects.get(id=holder.id)
            assert isinstance(fetched.job, UnknownSubclassCarrier)

            # Saving the parent after an unrelated change must not clobber
            # the unknown embedded subclass
            fetched.title = "t2"
            fetched.save()

            raw = get_as_pymongo(fetched)
            assert raw["job"]["_cls"] == "ScheduledCarrierJob"
            assert raw["job"]["cron"] == "* * * * *"
        finally:
            _DocumentRegistry.register(CarrierJob.ScheduledCarrierJob)
