import pytest
from bson import ObjectId, SON

from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentListField,
    FieldDoesNotExist,
    IntField,
    StringField,
    ValidationError,
)
from mongoengine.base.keyed import build_keyed_list_update
from mongoengine.connection import get_db
from tests.utils import MongoDBTestCase


try:
    import mongomock

    def _client_is_mongomock():
        return get_db().client.__class__.__module__.startswith("mongomock")

except ImportError:  # pragma: no cover
    def _client_is_mongomock():
        return False



class LineItem(EmbeddedDocument):
    sku = StringField()
    name = StringField()
    qty = IntField()


class Order(Document):
    items = EmbeddedDocumentListField(LineItem, key="sku")
    note = StringField()


def _make_order(items, note="note"):
    """Build an Order exactly like a document loaded from MongoDB."""
    son_items = [
        SON([("sku", item["sku"]), ("name", item["name"]), ("qty", item["qty"])])
        for item in items
    ]
    son = SON([("_id", ObjectId()), ("note", note), ("items", son_items)])
    order = Order._from_son(son)
    order._created = False
    return order


def _plan(order):
    items = order.items
    return build_keyed_list_update(
        field=Order._fields["items"],
        db_field="items",
        key_name="sku",
        current=items,
        snapshot=items._keyed_snapshot,
        snapshot_id=items._keyed_snapshot_id,
    )


class TestKeyedPlan:
    """Unit tests for the keyed diff; they don't need a live MongoDB."""

    BASE = [
        {"sku": "a", "name": "A", "qty": 1},
        {"sku": "b", "name": "B", "qty": 2},
        {"sku": "c", "name": "C", "qty": 3},
    ]

    def test_loaded_list_holds_a_snapshot(self):
        order = _make_order(self.BASE)
        assert order.items._keyed_key == "sku"
        assert [item["sku"] for item in order.items._keyed_snapshot] == ["a", "b", "c"]

    def test_no_change_is_a_noop(self):
        order = _make_order(self.BASE)
        plan = _plan(order)
        assert plan.safe is True
        assert plan.changed is False
        assert plan.update == {}

    def test_field_modification_targets_by_key(self):
        order = _make_order(self.BASE)
        order.items[1].qty = 20
        plan = _plan(order)
        assert plan.safe is True
        assert plan.update == {
            "$set": {"items.$[elitems1].qty": 20}
        }
        assert plan.array_filters == [{"elitems1.sku": "b"}]

    def test_unset_targets_by_key(self):
        order = _make_order(self.BASE)
        order.items[0]._mark_as_unset("name")
        plan = _plan(order)
        assert plan.safe is True
        assert plan.update["$unset"] == {"items.$[elitems0].name": 1}
        assert plan.array_filters == [{"elitems0.sku": "a"}]

    def test_delete_uses_keyed_pull(self):
        order = _make_order(self.BASE)
        order.items.remove(order.items[0])
        plan = _plan(order)
        assert plan.safe is True
        assert plan.update == {"$pull": {"items": {"sku": "a"}}}

    def test_delete_several_uses_pull_in(self):
        order = _make_order(self.BASE)
        order.items.remove(order.items[0])
        order.items.remove(order.items[0])  # removes "b"
        plan = _plan(order)
        assert plan.safe is True
        assert plan.update == {"$pull": {"items": {"sku": {"$in": ["a", "b"]}}}}

    def test_append_uses_tail_push(self):
        order = _make_order(self.BASE)
        order.items.create(sku="d", name="D", qty=4)
        plan = _plan(order)
        assert plan.safe is True
        assert list(plan.update) == ["$push"]
        pushed = plan.update["$push"]["items"]["$each"]
        assert [item["sku"] for item in pushed] == ["d"]

    def test_modify_and_delete_are_combined(self):
        order = _make_order(self.BASE)
        order.items[1].qty = 20
        order.items.remove(order.items[0])
        plan = _plan(order)
        assert plan.safe is True
        assert plan.update["$pull"] == {"items": {"sku": "a"}}
        assert plan.update["$set"] == {"items.$[elitems0].qty": 20}
        assert plan.array_filters == [{"elitems0.sku": "b"}]

    def test_modify_and_append_are_combined(self):
        order = _make_order(self.BASE)
        order.items[0].qty = 10
        order.items.create(sku="d")
        plan = _plan(order)
        assert plan.safe is True
        assert set(plan.update) == {"$set", "$push"}

    def test_reorder_falls_back_to_guarded_replacement(self):
        order = _make_order(self.BASE)
        order.items = [order.items[1], order.items[0], order.items[2]]
        plan = _plan(order)
        assert plan.safe is False
        assert [item["sku"] for item in plan.replacement] == ["b", "a", "c"]
        assert plan.condition is not None

    def test_reorder_with_modification_falls_back(self):
        order = _make_order(self.BASE)
        order.items = [order.items[2], order.items[0], order.items[1]]
        order.items[1].qty = 111
        plan = _plan(order)
        assert plan.safe is False
        assert plan.condition is not None

    def test_middle_insertion_falls_back(self):
        order = _make_order(self.BASE)
        order.items.insert(1, LineItem(sku="x", name="X", qty=0))
        plan = _plan(order)
        assert plan.safe is False
        assert plan.condition is not None

    def test_delete_and_append_falls_back(self):
        order = _make_order(self.BASE)
        order.items.remove(order.items[0])
        order.items.create(sku="d")
        plan = _plan(order)
        assert plan.safe is False
        assert plan.condition is not None

    def test_in_place_key_change_is_rejected(self):
        order = _make_order(self.BASE)
        order.items[0].sku = "zzz"
        with pytest.raises(ValidationError, match="cannot be changed in place"):
            _plan(order)

    def test_new_element_can_still_receive_its_key(self):
        order = _make_order(self.BASE)
        new_item = LineItem(name="D")
        order.items.append(new_item)
        new_item.sku = "d"
        plan = _plan(order)
        assert plan.safe is True
        assert "$push" in plan.update

    def test_duplicate_keys_are_rejected(self):
        order = _make_order(self.BASE)
        order.items[1].sku = "a"
        with pytest.raises(ValidationError, match="must be unique"):
            _plan(order)

    def test_two_copies_target_distinct_elements_after_reorder(self):
        # The scenario from the feature description: two copies modify
        # different elements and then the array order is changed.  The
        # generated update must keep targeting the *same business key*,
        # never the (now shifted) array position.
        copy_one = _make_order(self.BASE)
        copy_two = _make_order(self.BASE)

        copy_one.items[0].qty = 100  # modifies sku "a"
        copy_two.items[1].name = "B2"  # modifies sku "b"

        # Both copies reorder their view of the array.
        copy_one.items = [copy_one.items[2], copy_one.items[0], copy_one.items[1]]
        copy_two.items = [copy_two.items[2], copy_two.items[1], copy_two.items[0]]

        plan_one = _plan(copy_one)
        plan_two = _plan(copy_two)

        # Reordering + a field modification cannot be expressed safely, so
        # both copies fall back to the guarded replacement whose
        # precondition pins the persisted array to their common baseline.
        assert plan_one.safe is False
        assert plan_two.safe is False
        assert plan_one.condition is not None
        assert plan_two.condition is not None

    def test_field_modification_after_local_reorder_without_save_still_keyed(self):
        # Reordering inside the same list without reassignment keeps the
        # elements identified by key; the arrayFilter references the right
        # element even if its index moved.
        order = _make_order(self.BASE)
        order.items.reverse()  # [c, b, a] in place
        plan = _plan(order)
        assert plan.safe is False  # pure reorder -> guarded replacement

    def test_in_place_element_replacement_falls_back(self):
        order = _make_order(self.BASE)
        # A brand new object placed at an existing position but carrying the
        # same key cannot be diffed field by field.
        order.items[0] = LineItem(sku="a", name="REPLACED", qty=1)
        plan = _plan(order)
        assert plan.safe is False
        assert plan.condition is not None
        assert plan.replacement[0]["name"] == "REPLACED"

    def test_unknown_key_field_is_rejected_at_definition(self):
        class BadItem(EmbeddedDocument):
            name = StringField()

        with pytest.raises(ValueError, match="is not a field of"):
            class BadOrder(Document):
                items = EmbeddedDocumentListField(BadItem, key="missing")

    def test_lazy_key_validation_for_string_document_type(self):
        # A string (recursive) document type cannot be validated at
        # definition time; the plan builder validates it instead.
        class LazyOrder(Document):
            items = EmbeddedDocumentListField("LineItem", key="missing")

        order = LazyOrder._from_son(
            SON([("_id", ObjectId()), ("items", [SON([("sku", "x")])])])
        )
        with pytest.raises(FieldDoesNotExist):
            build_keyed_list_update(
                field=LazyOrder._fields["items"],
                db_field="items",
                key_name="missing",
                current=order.items,
                snapshot=order.items._keyed_snapshot,
                snapshot_id=order.items._keyed_snapshot_id,
            )



class TestKeyedEmbeddedDocumentListField(MongoDBTestCase):
    def setUp(self):
        Order.drop_collection()

    def _insert(self, items):
        order = Order(items=[LineItem(**item) for item in items])
        order.save()
        return Order.objects.get(id=order.id)

    def test_snapshot_refreshed_after_save(self):
        order = self._insert(
            [{"sku": "a", "name": "A", "qty": 1}]
        )
        assert [item["sku"] for item in order.items._keyed_snapshot] == ["a"]

    def test_modification_generates_array_filters_update(self):
        if _client_is_mongomock():
            pytest.skip("mongomock does not implement arrayFilters")
        order = self._insert(
            [
                {"sku": "a", "name": "A", "qty": 1},
                {"sku": "b", "name": "B", "qty": 2},
            ]
        )
        # Simulate a concurrent reindexing of the array by another process.
        Order.objects(
            id=order.id
        ).update(pull__items={"sku": "a"})

        order.items[1].qty = 20  # targets sku "b", now at index 0 in DB
        order.save()

        raw = Order.objects.as_pymongo().get(id=order.id)
        assert raw["items"] == [{"sku": "b", "name": "B", "qty": 20}]

    def test_delete_uses_keyed_pull(self):
        order = self._insert(
            [
                {"sku": "a", "name": "A", "qty": 1},
                {"sku": "b", "name": "B", "qty": 2},
            ]
        )
        order.items.remove(order.items[0])
        order.save()
        raw = Order.objects.as_pymongo().get(id=order.id)
        assert [item["sku"] for item in raw["items"]] == ["b"]

    def test_reorder_guarded_replacement_raises_on_concurrent_change(self):
        from mongoengine.errors import SaveConditionError

        order = self._insert(
            [
                {"sku": "a", "name": "A", "qty": 1},
                {"sku": "b", "name": "B", "qty": 2},
            ]
        )
        order.items = [order.items[1], order.items[0]]

        # Another process modifies the array concurrently.
        Order.objects(id=order.id).update(
            push__items={"sku": "z", "name": "Z", "qty": 0}
        )

        with pytest.raises(SaveConditionError):
            order.save()

    def test_in_place_key_change_is_rejected_on_save(self):
        order = self._insert([{"sku": "a", "name": "A", "qty": 1}])
        order.items[0].sku = "b"
        with pytest.raises(ValidationError, match="cannot be changed in place"):
            order.save()

    def test_two_copies_modify_different_elements(self):
        if _client_is_mongomock():
            pytest.skip("mongomock does not implement arrayFilters")
        order = self._insert(
            [
                {"sku": "a", "name": "A", "qty": 1},
                {"sku": "b", "name": "B", "qty": 2},
            ]
        )
        copy_one = Order.objects.get(id=order.id)
        copy_two = Order.objects.get(id=order.id)
        copy_one.items.get(sku="a").qty = 10
        copy_two.items.get(sku="b").qty = 20
        copy_one.save()
        copy_two.save()

        raw = Order.objects.as_pymongo().get(id=order.id)
        assert {item["sku"]: item["qty"] for item in raw["items"]} == {
            "a": 10,
            "b": 20,
        }
