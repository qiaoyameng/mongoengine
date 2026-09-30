import weakref
from copy import deepcopy

import pytest
from bson import ObjectId

from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    EmbeddedDocumentListField,
    GenericEmbeddedDocumentField,
    InvalidDocumentError,
    IntField,
    InvalidQueryError,
    ListField,
    LookUpError,
    MapField,
    StringField,
    ValidationError,
)
from tests.utils import MongoDBTestCase


class TestEmbeddedDocumentField(MongoDBTestCase):
    def test___init___(self):
        class MyDoc(EmbeddedDocument):
            name = StringField()

        field = EmbeddedDocumentField(MyDoc)
        assert field.document_type_obj == MyDoc

        field2 = EmbeddedDocumentField("MyDoc")
        assert field2.document_type_obj == "MyDoc"

    def test___init___throw_error_if_document_type_is_not_EmbeddedDocument(self):
        with pytest.raises(ValidationError):
            EmbeddedDocumentField(dict)

    def test_document_type_throw_error_if_not_EmbeddedDocument_subclass(self):
        class MyDoc(Document):
            name = StringField()

        emb = EmbeddedDocumentField("MyDoc")
        with pytest.raises(ValidationError) as exc_info:
            emb.document_type
        assert (
            "Invalid embedded document class provided to an EmbeddedDocumentField"
            in str(exc_info.value)
        )

    def test_embedded_document_field_only_allow_subclasses_of_embedded_document(self):
        # Relates to #1661
        class MyDoc(Document):
            name = StringField()

        with pytest.raises(ValidationError):

            class MyFailingDoc(Document):
                emb = EmbeddedDocumentField(MyDoc)

        with pytest.raises(ValidationError):

            class MyFailingdoc2(Document):
                emb = EmbeddedDocumentField("MyDoc")

    def test_embedded_document_list_field__has__instance_weakref(self):
        class Comment(EmbeddedDocument):
            content = StringField()

        class Post(Document):
            title = StringField()
            comment = EmbeddedDocumentField(Comment)
            comments = EmbeddedDocumentListField(Comment)
            comments2 = ListField(EmbeddedDocumentField(Comment))

        Post.drop_collection()

        for i in range(5):
            Post(
                title=f"{i}",
                comment=Comment(content=f"{i}"),
                comments=[Comment(content=f"{i}")],
                comments2=[Comment(content=f"{i}")],
            ).save()

        posts = list(Post.objects)
        for post in posts:
            assert isinstance(post.comments._instance, weakref.ProxyTypes)
            assert isinstance(post.comments2._instance, weakref.ProxyTypes)
            assert isinstance(post.comment._instance, weakref.ProxyTypes)
            for comment in post.comments:
                assert isinstance(comment._instance, weakref.ProxyTypes)
            for comment2 in post.comments2:
                assert isinstance(comment2._instance, weakref.ProxyTypes)

    def test_embedded_document_field_validate_subclass(self):
        class BaseItem(EmbeddedDocument):
            f = IntField()

            meta = {"allow_inheritance": True}

            def validate(self, clean=True):
                if self.f == 0:
                    raise Exception("can not be 0")
                return super().validate(clean)

        class RealItem(BaseItem):
            a = IntField()

            def validate(self, clean=True):
                if self.f == 1:
                    raise Exception("can not be 1")
                return super().validate(clean)

        class TopLevel(Document):
            item = EmbeddedDocumentField(document_type=BaseItem)
            items = EmbeddedDocumentListField(document_type=BaseItem)

        passing_item = RealItem(f=2, a=0)
        item = TopLevel(item=passing_item, items=[passing_item])
        item.validate()

        failing_item = RealItem(f=1, a=0)
        item = TopLevel(item=failing_item)
        with pytest.raises(Exception, match="can not be 1"):
            item.validate()

        item = TopLevel(items=[failing_item])
        with pytest.raises(Exception, match="can not be 1"):
            item.validate()

        # verify that super calls the parent
        failing_item_in_base = RealItem(f=0, a=0)
        item = TopLevel(item=failing_item_in_base)
        with pytest.raises(Exception, match="can not be 0"):
            item.validate()

    def test_query_embedded_document_attribute(self):
        class AdminSettings(EmbeddedDocument):
            foo1 = StringField()
            foo2 = StringField()

        class Person(Document):
            settings = EmbeddedDocumentField(AdminSettings)
            name = StringField()

        Person.drop_collection()

        p = Person(settings=AdminSettings(foo1="bar1", foo2="bar2"), name="John").save()

        # Test non exiting attribute
        with pytest.raises(InvalidQueryError) as exc_info:
            Person.objects(settings__notexist="bar").first()
        assert str(exc_info.value) == 'Cannot resolve field "notexist"'

        with pytest.raises(LookUpError):
            Person.objects.only("settings.notexist")

        # Test existing attribute
        assert Person.objects(settings__foo1="bar1").first().id == p.id
        only_p = Person.objects.only("settings.foo1").first()
        assert only_p.settings.foo1 == p.settings.foo1
        assert only_p.settings.foo2 is None
        assert only_p.name is None

        exclude_p = Person.objects.exclude("settings.foo1").first()
        assert exclude_p.settings.foo1 is None
        assert exclude_p.settings.foo2 == p.settings.foo2
        assert exclude_p.name == p.name

    def test_query_embedded_document_attribute_with_inheritance(self):
        class BaseSettings(EmbeddedDocument):
            meta = {"allow_inheritance": True}
            base_foo = StringField()

        class AdminSettings(BaseSettings):
            sub_foo = StringField()

        class Person(Document):
            settings = EmbeddedDocumentField(BaseSettings)

        Person.drop_collection()

        p = Person(settings=AdminSettings(base_foo="basefoo", sub_foo="subfoo"))
        p.save()

        # Test non exiting attribute
        with pytest.raises(InvalidQueryError) as exc_info:
            assert Person.objects(settings__notexist="bar").first().id == p.id
        assert str(exc_info.value) == 'Cannot resolve field "notexist"'

        # Test existing attribute
        assert Person.objects(settings__base_foo="basefoo").first().id == p.id
        assert Person.objects(settings__sub_foo="subfoo").first().id == p.id

        only_p = Person.objects.only("settings.base_foo", "settings._cls").first()
        assert only_p.settings.base_foo == "basefoo"
        assert only_p.settings.sub_foo is None

    def test_query_list_embedded_document_with_inheritance(self):
        class Post(EmbeddedDocument):
            title = StringField(max_length=120, required=True)
            meta = {"allow_inheritance": True}

        class TextPost(Post):
            content = StringField()

        class MoviePost(Post):
            author = StringField()

        class Record(Document):
            posts = ListField(EmbeddedDocumentField(Post))

        record_movie = Record(posts=[MoviePost(author="John", title="foo")]).save()
        record_text = Record(posts=[TextPost(content="a", title="foo")]).save()

        records = list(Record.objects(posts__author=record_movie.posts[0].author))
        assert len(records) == 1
        assert records[0].id == record_movie.id

        records = list(Record.objects(posts__content=record_text.posts[0].content))
        assert len(records) == 1
        assert records[0].id == record_text.id

        assert Record.objects(posts__title="foo").count() == 2


class TestGenericEmbeddedDocumentField(MongoDBTestCase):
    def test_generic_embedded_document(self):
        class Car(EmbeddedDocument):
            name = StringField()

        class Dish(EmbeddedDocument):
            food = StringField(required=True)
            number = IntField()

        class Person(Document):
            name = StringField()
            like = GenericEmbeddedDocumentField()

        Person.drop_collection()

        person = Person(name="Test User")
        person.like = Car(name="Fiat")
        person.save()

        person = Person.objects.first()
        assert isinstance(person.like, Car)

        person.like = Dish(food="arroz", number=15)
        person.save()

        person = Person.objects.first()
        assert isinstance(person.like, Dish)

    def test_generic_embedded_document_choices(self):
        """Ensure you can limit GenericEmbeddedDocument choices."""

        class Car(EmbeddedDocument):
            name = StringField()

        class Dish(EmbeddedDocument):
            food = StringField(required=True)
            number = IntField()

        class Person(Document):
            name = StringField()
            like = GenericEmbeddedDocumentField(choices=(Dish,))

        Person.drop_collection()

        person = Person(name="Test User")
        person.like = Car(name="Fiat")
        with pytest.raises(ValidationError):
            person.validate()

        person.like = Dish(food="arroz", number=15)
        person.save()

        person = Person.objects.first()
        assert isinstance(person.like, Dish)

    def test_generic_embedded_document_choices_accept_mongo_dict(self):
        class Dish(EmbeddedDocument):
            food = StringField()

        field = GenericEmbeddedDocumentField(choices=(Dish,))
        mongo_value = field.to_mongo(Dish(food="arroz"))

        assert type(mongo_value) is dict
        assert field.validate(mongo_value) is True

    def test_generic_list_embedded_document_choices(self):
        """Ensure you can limit GenericEmbeddedDocument choices inside
        a list field.
        """

        class Car(EmbeddedDocument):
            name = StringField()

        class Dish(EmbeddedDocument):
            food = StringField(required=True)
            number = IntField()

        class Person(Document):
            name = StringField()
            likes = ListField(GenericEmbeddedDocumentField(choices=(Dish,)))

        Person.drop_collection()

        person = Person(name="Test User")
        person.likes = [Car(name="Fiat")]
        with pytest.raises(ValidationError):
            person.validate()

        person.likes = [Dish(food="arroz", number=15)]
        person.save()

        person = Person.objects.first()
        assert isinstance(person.likes[0], Dish)

    def test_choices_validation_documents(self):
        """
        Ensure fields with document choices validate given a valid choice.
        """

        class UserComments(EmbeddedDocument):
            author = StringField()
            message = StringField()

        class BlogPost(Document):
            comments = ListField(GenericEmbeddedDocumentField(choices=(UserComments,)))

        # Ensure Validation Passes
        BlogPost(comments=[UserComments(author="user2", message="message2")]).save()

    def test_choices_validation_documents_invalid(self):
        """
        Ensure fields with document choices validate given an invalid choice.
        This should throw a ValidationError exception.
        """

        class UserComments(EmbeddedDocument):
            author = StringField()
            message = StringField()

        class ModeratorComments(EmbeddedDocument):
            author = StringField()
            message = StringField()

        class BlogPost(Document):
            comments = ListField(GenericEmbeddedDocumentField(choices=(UserComments,)))

        # Single Entry Failure
        post = BlogPost(comments=[ModeratorComments(author="mod1", message="message1")])
        with pytest.raises(ValidationError):
            post.save()

        # Mixed Entry Failure
        post = BlogPost(
            comments=[
                ModeratorComments(author="mod1", message="message1"),
                UserComments(author="user2", message="message2"),
            ]
        )
        with pytest.raises(ValidationError):
            post.save()

    def test_choices_validation_documents_inheritance(self):
        """
        Ensure fields with document choices validate given subclass of choice.
        """

        class Comments(EmbeddedDocument):
            meta = {"abstract": True}
            author = StringField()
            message = StringField()

        class UserComments(Comments):
            pass

        class BlogPost(Document):
            comments = ListField(GenericEmbeddedDocumentField(choices=(Comments,)))

        # Save Valid EmbeddedDocument Type
        BlogPost(comments=[UserComments(author="user2", message="message2")]).save()

    def test_query_generic_embedded_document_attribute(self):
        class AdminSettings(EmbeddedDocument):
            foo1 = StringField()

        class NonAdminSettings(EmbeddedDocument):
            foo2 = StringField()

        class Person(Document):
            settings = GenericEmbeddedDocumentField(
                choices=(AdminSettings, NonAdminSettings)
            )

        Person.drop_collection()

        p1 = Person(settings=AdminSettings(foo1="bar1")).save()
        p2 = Person(settings=NonAdminSettings(foo2="bar2")).save()

        # Test non exiting attribute
        with pytest.raises(InvalidQueryError) as exc_info:
            Person.objects(settings__notexist="bar").first()
        assert str(exc_info.value) == 'Cannot resolve field "notexist"'

        with pytest.raises(LookUpError):
            Person.objects.only("settings.notexist")

        # Test existing attribute
        assert Person.objects(settings__foo1="bar1").first().id == p1.id
        assert Person.objects(settings__foo2="bar2").first().id == p2.id

    def test_query_generic_embedded_document_attribute_with_inheritance(self):
        class BaseSettings(EmbeddedDocument):
            meta = {"allow_inheritance": True}
            base_foo = StringField()

        class AdminSettings(BaseSettings):
            sub_foo = StringField()

        class Person(Document):
            settings = GenericEmbeddedDocumentField(choices=[BaseSettings])

        Person.drop_collection()

        p = Person(settings=AdminSettings(base_foo="basefoo", sub_foo="subfoo"))
        p.save()

        # Test non exiting attribute
        with pytest.raises(InvalidQueryError) as exc_info:
            assert Person.objects(settings__notexist="bar").first().id == p.id
        assert str(exc_info.value) == 'Cannot resolve field "notexist"'

        # Test existing attribute
        assert Person.objects(settings__base_foo="basefoo").first().id == p.id
        assert Person.objects(settings__sub_foo="subfoo").first().id == p.id

    def test_deepcopy_set__instance(self):
        """Ensure that the _instance attribute on EmbeddedDocument exists after a deepcopy"""

        class Wallet(EmbeddedDocument):
            money = IntField()

        class Person(Document):
            wallet = EmbeddedDocumentField(Wallet)
            wallet_map = MapField(EmbeddedDocumentField(Wallet))

        # Test on fresh EmbeddedDoc
        emb_doc = Wallet(money=1)
        assert emb_doc._instance is None
        copied_emb_doc = deepcopy(emb_doc)
        assert copied_emb_doc._instance is None

        # Test on attached EmbeddedDoc
        doc = Person(
            id=ObjectId(), wallet=Wallet(money=2), wallet_map={"test": Wallet(money=2)}
        )
        assert doc.wallet._instance == doc
        copied_emb_doc = deepcopy(doc.wallet)
        assert copied_emb_doc._instance is None

        copied_map_emb_doc = deepcopy(doc.wallet_map)
        assert copied_map_emb_doc["test"]._instance is None


class TestKeyedEmbeddedDocumentListField:
    def test_keyed_mode_is_explicit(self):
        class KeyItem(EmbeddedDocument):
            key = StringField()
            name = StringField()

        default_field = EmbeddedDocumentListField(KeyItem)
        assert default_field.keyed is False
        assert default_field.key_field is None

        implicit_field = EmbeddedDocumentListField(KeyItem, key_field="name")
        assert implicit_field.keyed is True
        assert implicit_field.key_field == "name"

        unknown_field = EmbeddedDocumentListField(KeyItem, keyed=True)
        unknown_field.key_field = "missing"
        with pytest.raises(InvalidQueryError, match="Unknown key field"):
            unknown_field.validate([KeyItem(key="a")])

    @staticmethod
    def _document():
        class Item(EmbeddedDocument):
            business_key = StringField(db_field="bk")
            name = StringField()

        class KeyedDoc(Document):
            title = StringField()
            items = EmbeddedDocumentListField(
                Item, keyed=True, key_field="business_key", db_field="items"
            )

        raw = {
            "_id": ObjectId(),
            "title": "post",
            "items": [
                {"bk": "a", "name": "A"},
                {"bk": "b", "name": "B"},
            ],
        }
        return Item, KeyedDoc, KeyedDoc._from_son(raw, created=False), raw

    def test_field_modifications_use_keyed_array_filters(self):
        Item, KeyedDoc, doc, raw = self._document()
        doc.items[0].name = "A2"
        doc.items[1].name = "B2"

        update, _, _, array_filters, is_fallback = doc._get_save_update_plan()

        assert update == {
            "$set": {
                "items.$[item_0].name": "A2",
                "items.$[item_1].name": "B2",
            }
        }
        assert array_filters == [
            {"item_0.bk": "a"},
            {"item_1.bk": "b"},
        ]

    def test_pull_and_append_do_not_use_indices(self):
        Item, KeyedDoc, doc, raw = self._document()
        doc.items.pop(0)
        update, _, precondition, _, is_fallback = doc._get_save_update_plan()
        assert update == {
            "$pull": {"items": {"bk": "a"}}
        }
        assert precondition == {"items": {"$exists": True, "$type": "array"}}
        assert is_fallback is False

        Item, KeyedDoc, doc, raw = self._document()
        doc.items.append(Item(business_key="c", name="C"))
        update, _, precondition, _, is_fallback = doc._get_save_update_plan()
        assert update == {
            "$push": {"items": {"bk": "c", "name": "C"}}
        }
        assert precondition == {
            "items": {
                "$exists": True,
                "$type": "array",
                "$not": {"$elemMatch": {"bk": "c"}},
            }
        }
        assert is_fallback is False

    def test_reorder_or_unsafe_combination_uses_preconditioned_full_replace(self):
        Item, KeyedDoc, doc, raw = self._document()
        doc.items.reverse()

        update, fallback, precondition, array_filters, is_fallback = (
            doc._get_save_update_plan()
        )
        assert is_fallback is True

        assert update == {}
        assert fallback == {
            "$set": {
                "items": [
                    {"bk": "b", "name": "B"},
                    {"bk": "a", "name": "A"},
                ]
            }
        }
        assert precondition == {"items": raw["items"]}
        assert array_filters == []

        Item, KeyedDoc, doc, raw = self._document()
        doc.items[0].name = "A2"
        doc.items.pop(1)
        _, fallback, precondition, array_filters, is_fallback = (
            doc._get_save_update_plan()
        )
        assert fallback == {
            "$set": {"items": [{"bk": "a", "name": "A2"}]}
        }
        assert precondition == {"items": raw["items"]}
        assert array_filters == []

    def test_in_place_key_change_is_rejected(self):
        Item, KeyedDoc, doc, raw = self._document()
        doc.items[0].business_key = "c"

        with pytest.raises(InvalidDocumentError, match="immutable"):
            doc._get_save_update_plan()

    def test_duplicate_keys_are_rejected(self):
        Item, KeyedDoc, doc, raw = self._document()
        doc.items.append(Item(business_key="a", name="Duplicate"))

        with pytest.raises(ValidationError, match="unique"):
            doc.validate()

    def test_two_copies_reorder_and_still_target_by_business_key(self):
        Item, KeyedDoc, copy_a, raw = self._document()
        _, _, copy_b, _ = self._document()

        copy_a.items[0].name = "A2"
        copy_a.items.insert(0, copy_a.items.pop(1))
        _, fallback, precondition, array_filters, is_fallback = (
            copy_a._get_save_update_plan()
        )
        assert precondition == {"items": raw["items"]}
        assert fallback["$set"]["items"] == [
            {"bk": "b", "name": "B"},
            {"bk": "a", "name": "A2"},
        ]
        assert array_filters == []

        copy_b.items[1].name = "B2"
        update, _, precondition, array_filters, is_fallback = (
            copy_b._get_save_update_plan()
        )
        assert precondition == {"items": {"$elemMatch": {"bk": "b"}}}
        assert is_fallback is False
        assert update == {"$set": {"items.$[item_0].name": "B2"}}
        assert array_filters == [{"item_0.bk": "b"}]

    def test_nested_keyed_list_uses_absolute_array_filter_path(self):
        class NestedItem(EmbeddedDocument):
            nested_key = StringField(db_field="nk")
            name = StringField()

        class Container(EmbeddedDocument):
            nested_items = EmbeddedDocumentListField(
                NestedItem, keyed=True, key_field="nested_key", db_field="nested"
            )

        class NestedDoc(Document):
            container = EmbeddedDocumentField(Container)

        raw = {
            "_id": ObjectId(),
            "container": {
                "nested": [
                    {"nk": "a", "name": "A"},
                    {"nk": "b", "name": "B"},
                ]
            },
        }
        doc = NestedDoc._from_son(raw, created=False)
        doc.container.nested_items[1].name = "B2"

        update, _, precondition, array_filters, is_fallback = (
            doc._get_save_update_plan()
        )

        assert is_fallback is False
        assert update == {
            "$set": {"container.nested.$[item_0].name": "B2"}
        }
        assert precondition == {
            "container.nested": {"$elemMatch": {"nk": "b"}}
        }
        assert array_filters == [{"item_0.nk": "b"}]

    def test_save_passes_array_filters_for_safe_update(self, monkeypatch):
        class UpdateResult:
            def __init__(self):
                self.raw_result = {"n": 1, "updatedExisting": True}

        class FakeCollection:
            write_concern = type("WriteConcern", (), {"document": {}})()

            def __init__(self):
                self.calls = []

            def with_options(self, **kwargs):
                return self

            def update_one(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return UpdateResult()

        Item, KeyedDoc, doc, raw = self._document()
        doc.items[1].name = "B2"
        collection = FakeCollection()
        monkeypatch.setattr(
            KeyedDoc, "_get_collection", lambda self: collection
        )

        doc._save_update(doc.to_mongo(), None, {})

        args, kwargs = collection.calls[0]
        assert args[0] == {
            "_id": raw["_id"],
            "items": {"$elemMatch": {"bk": "b"}},
        }
        assert args[1] == {"$set": {"items.$[item_0].name": "B2"}}
        assert kwargs["array_filters"] == [{"item_0.bk": "b"}]

    def test_save_fallback_uses_old_array_precondition(self, monkeypatch):
        class UpdateResult:
            def __init__(self):
                self.raw_result = {"n": 1, "updatedExisting": True}

        class FakeCollection:
            write_concern = type("WriteConcern", (), {"document": {}})()

            def __init__(self):
                self.calls = []

            def with_options(self, **kwargs):
                return self

            def update_one(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return UpdateResult()

        Item, KeyedDoc, doc, raw = self._document()
        doc.items.reverse()
        collection = FakeCollection()
        monkeypatch.setattr(
            KeyedDoc, "_get_collection", lambda self: collection
        )

        doc._save_update(doc.to_mongo(), None, {})

        args, kwargs = collection.calls[0]
        assert args[0]["_id"] == raw["_id"]
        assert args[0]["items"] == raw["items"]
        assert args[1] == {
            "$set": {
                "items": [
                    {"bk": "b", "name": "B"},
                    {"bk": "a", "name": "A"},
                ]
            }
        }
        assert kwargs["array_filters"] is None
        assert kwargs["upsert"] is False
