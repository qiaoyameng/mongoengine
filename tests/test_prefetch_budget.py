from bson import DBRef, ObjectId
import pytest

from mongoengine import (
    Document,
    GenericReferenceField,
    LazyReferenceField,
    ListField,
    PrefetchBudget,
    ReferenceField,
    StringField,
)
from mongoengine.base.datastructures import (
    BudgetedLazyReference,
    GenericBudgetedLazyReference,
)
from mongoengine.dereference import DeReference
from mongoengine.queryset import QuerySet


class CountableStore:
    def __init__(self):
        self.data = {}
        self.calls = {}

    def add(self, document):
        collection = document._get_collection_name()
        self.data.setdefault(collection, {})[document.pk] = document
        self.calls.setdefault(collection, [])

    def queryset(self, document_type):
        store = self

        class CountableQuerySet:
            def in_bulk(self, ids):
                collection = document_type._get_collection_name()
                store.calls.setdefault(collection, []).append(list(ids))
                return {
                    object_id: store.data[collection][object_id]
                    for object_id in ids
                    if object_id in store.data.get(collection, {})
                }

        return CountableQuerySet()

    def patch(self, *document_types):
        for document_type in document_types:
            store = self
            document_type.objects = classmethod(
                lambda cls, store=store: store.queryset(cls)
            )

    @property
    def query_count(self):
        return sum(len(calls) for calls in self.calls.values())


class SharedTargetDocument(Document):
    meta = {"collection": "prefetch_shared_target"}

    name = StringField()


class SharedRootDocument(Document):
    meta = {"collection": "prefetch_shared_root"}

    first = ReferenceField(SharedTargetDocument)
    second = ReferenceField(SharedTargetDocument)
    items = ListField(ReferenceField(SharedTargetDocument))


class CycleNodeDocument(Document):
    meta = {"collection": "prefetch_cycle_node"}

    name = StringField()
    next = ReferenceField("self")
    alternatives = ListField(ReferenceField("self"))


class CrossRootDocument(Document):
    meta = {"collection": "prefetch_cross_root"}

    link = ReferenceField("CrossTargetDocument")


class CrossTargetDocument(Document):
    meta = {"collection": "prefetch_cross_target"}

    name = StringField()


class OtherTargetDocument(Document):
    meta = {"collection": "prefetch_cross_other_target"}

    name = StringField()


class CrossManyRootDocument(Document):
    meta = {"collection": "prefetch_cross_many_root"}

    first = ReferenceField(CrossTargetDocument)
    second = ReferenceField(OtherTargetDocument)


class GenericRootDocument(Document):
    meta = {"collection": "prefetch_generic_root"}

    link = GenericReferenceField()


class ExplicitLazyRootDocument(Document):
    meta = {"collection": "prefetch_explicit_lazy_root"}

    link = LazyReferenceField(SharedTargetDocument)


def test_shared_reference_is_counted_and_fetched_once():
    target_id = ObjectId()
    target = SharedTargetDocument(id=target_id, name="shared")
    root = SharedRootDocument(
        id=ObjectId(), first=target_id, second=target_id, items=[target_id]
    )
    store = CountableStore()
    store.add(target)
    store.patch(SharedTargetDocument)

    _, report = DeReference()(
        [root], budget=PrefetchBudget(max_depth=1)
    )

    assert report.queries_executed == 1
    assert report.unique_objects_fetched == 1
    assert report.complete is True
    assert len(store.calls[SharedTargetDocument._get_collection_name()]) == 1
    assert root.first is target
    assert root.second is target
    assert root.items == [target]


def test_unique_object_budget_leaves_identifiable_lazy_references():
    target_ids = [ObjectId(), ObjectId(), ObjectId()]
    targets = [
        SharedTargetDocument(id=target_id, name=str(index))
        for index, target_id in enumerate(target_ids)
    ]
    root = SharedRootDocument(
        id=ObjectId(),
        first=target_ids[0],
        second=target_ids[1],
        items=[target_ids[2]],
    )
    store = CountableStore()
    for target in targets:
        store.add(target)
    store.patch(SharedTargetDocument)

    _, report = DeReference()(
        [root],
        budget=PrefetchBudget(
            max_depth=1, max_unique_objects=1, max_queries_per_round=1
        ),
    )

    assert report.truncated_by == "max_unique_objects"
    assert report.unique_objects_fetched == 1
    assert report.queries_executed == 1
    assert report.unresolved_paths == ("[0].second", "[0].items[0]")
    assert isinstance(root.second, BudgetedLazyReference)
    assert isinstance(root.items[0], BudgetedLazyReference)
    assert root.second.passthrough is False
    assert "BudgetedLazyReference" in repr(root.second)
    assert str(target_ids[1]) in root.to_json()
    with pytest.raises(AttributeError):
        root.second.name


def test_cycle_is_traversed_only_once_per_object():
    first_id = ObjectId()
    second_id = ObjectId()
    third_id = ObjectId()
    first = CycleNodeDocument(id=first_id, name="first")
    second = CycleNodeDocument(id=second_id, name="second")
    third = CycleNodeDocument(id=third_id, name="third")
    first._data["next"] = second_id
    second._data["next"] = first_id
    second._data["alternatives"] = [first_id, third_id]

    store = CountableStore()
    store.add(first)
    store.add(second)
    store.add(third)
    store.patch(CycleNodeDocument)

    _, report = DeReference()(
        [first], budget=PrefetchBudget(max_depth=1)
    )

    assert report.queries_executed == 1
    assert report.unique_objects_fetched == 1
    assert report.complete is False
    assert report.unresolved[0].path == "[0].next.alternatives[1]"
    assert report.unresolved[0].reason == "max_depth"
    assert first.next is second
    assert second.next is first
    assert second.alternatives[0] is first
    assert isinstance(second.alternatives[1], BudgetedLazyReference)


def test_cross_collections_have_separate_batched_queries():
    target_id = ObjectId()
    root_id = ObjectId()
    target = CrossTargetDocument(id=target_id, name="target")
    root = CrossRootDocument(id=root_id, link=target_id)

    store = CountableStore()
    store.add(target)
    store.patch(CrossTargetDocument)

    _, report = DeReference()(
        [root],
        budget=PrefetchBudget(max_depth=1, max_queries_per_round=0),
    )

    assert report.queries_executed == 0
    assert report.truncated_by == "max_queries_per_round"
    assert report.unresolved_paths == ("[0].link",)
    assert isinstance(root.link, BudgetedLazyReference)
    assert root.link.reason == "max_queries_per_round"
    assert root.link.document_type is CrossTargetDocument


def test_query_budget_limits_cross_collection_rounds():
    target_id = ObjectId()
    other_id = ObjectId()
    target = CrossTargetDocument(id=target_id, name="target")
    other = OtherTargetDocument(id=other_id, name="other")
    root = CrossManyRootDocument(
        id=ObjectId(), first=target_id, second=other_id
    )
    store = CountableStore()
    store.add(target)
    store.add(other)
    store.patch(CrossTargetDocument, OtherTargetDocument)

    _, report = DeReference()(
        [root],
        budget=PrefetchBudget(max_depth=1, max_queries_per_round=1),
    )

    assert report.queries_executed == 1
    assert report.truncated_by == "max_queries_per_round"
    assert len(report.unresolved_paths) == 1
    resolved = [value for value in (root.first, root.second) if not isinstance(
        value, BudgetedLazyReference
    )]
    unresolved = [
        value
        for value in (root.first, root.second)
        if isinstance(value, BudgetedLazyReference)
    ]
    assert resolved == [other]
    assert len(unresolved) == 1


def test_generic_reference_is_kept_lazy_and_serializes_safely():
    target_id = ObjectId()
    target = CrossTargetDocument(id=target_id, name="generic-target")
    root = GenericRootDocument(id=ObjectId())
    root._data["link"] = {
        "_cls": "CrossTargetDocument",
        "_ref": DBRef(CrossTargetDocument._get_collection_name(), target_id),
    }

    _, report = DeReference()(
        [root],
        budget=PrefetchBudget(max_depth=1, max_queries_per_round=0),
    )

    assert isinstance(root._data["link"], BudgetedLazyReference)
    assert report.unresolved[0].document_class == "CrossTargetDocument"
    serialized = root.to_json()
    assert str(target_id) in serialized
    assert "generic-target" not in serialized
    with pytest.raises(AttributeError):
        root.link.name


def test_unknown_generic_collection_uses_safe_generic_placeholder():
    target_id = ObjectId()
    root = GenericRootDocument(
        id=ObjectId(),
        link=DBRef("prefetch_unknown_collection", target_id),
    )

    _, report = DeReference()(
        [root], budget=PrefetchBudget(max_depth=1)
    )

    assert isinstance(root.link, GenericBudgetedLazyReference)
    assert report.unresolved[0].collection == "prefetch_unknown_collection"
    assert report.unresolved[0].reason == "unknown_document_type"
    assert "GenericBudgetedLazyReference" in repr(root.link)
    assert str(target_id) in root.to_json()


def test_existing_lazy_reference_is_not_prefetched():
    target_id = ObjectId()
    target = SharedTargetDocument(id=target_id, name="explicit-lazy")
    root = ExplicitLazyRootDocument(id=ObjectId(), link=target_id)
    store = CountableStore()
    store.add(target)
    store.patch(SharedTargetDocument)

    _, report = DeReference()(
        [root], budget=PrefetchBudget(max_depth=3)
    )

    assert report.queries_executed == 0
    assert report.unique_objects_fetched == 0
    assert report.complete is True
    assert root.link.pk == target_id


def test_queryset_select_related_returns_queryset_and_report(monkeypatch):
    target_id = ObjectId()
    target = CrossTargetDocument(id=target_id, name="target")
    root = CrossRootDocument(id=ObjectId(), link=target_id)
    store = CountableStore()
    store.add(target)
    store.patch(CrossTargetDocument)

    class TestQuerySet(QuerySet):
        def __iter__(self):
            return iter([root])

    queryset = TestQuerySet(CrossRootDocument, object())
    result, report = queryset.select_related(
        max_depth=1, return_report=True
    )

    result._result_cache = [root]
    result._has_more = False
    assert result is not queryset
    assert list(result) == [root]
    assert root.link is target
    assert report.unique_objects_fetched == 1
    assert report.queries_executed == 1


def test_document_select_related_returns_self_and_report():
    target_id = ObjectId()
    target = CrossTargetDocument(id=target_id, name="target")
    root = CrossRootDocument(id=ObjectId(), link=target_id)
    store = CountableStore()
    store.add(target)
    store.patch(CrossTargetDocument)

    result, report = root.select_related(return_report=True)

    assert result is root
    assert root.link is target
    assert report.complete is True
    assert report.unique_objects_fetched == 1


def test_document_select_related_budget_keeps_lazy_reference():
    target_id = ObjectId()
    root = CrossRootDocument(id=ObjectId(), link=target_id)
    store = CountableStore()
    store.patch(CrossTargetDocument)

    result, report = root.select_related(
        max_queries_per_round=0, return_report=True
    )

    assert result is root
    assert isinstance(root.link, BudgetedLazyReference)
    assert report.truncated_by == "max_queries_per_round"
    assert report.unresolved_paths == ("link",)
