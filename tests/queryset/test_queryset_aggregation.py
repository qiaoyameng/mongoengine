import copy
import unittest
from unittest.mock import patch

import pytest
from pymongo.read_preferences import ReadPreference

from bson import encode

from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    IntField,
    ListField,
    PointField,
    StringField,
)
from mongoengine.errors import InvalidQueryError
from mongoengine.queryset.aggregation import transform_pipeline
from mongoengine.mongodb_support import (
    MONGODB_36,
    get_mongodb_version,
)
from tests.utils import MongoDBTestCase, db_ops_tracker


class TestAggregationModelFields(unittest.TestCase):

    def test_pipeline_conversion_and_non_mutation(self):
        class AggComment(EmbeddedDocument):
            author_name = StringField(db_field="authorName")

        class AggPost(Document):
            title = StringField(db_field="postTitle")
            comments = ListField(
                EmbeddedDocumentField(AggComment), db_field="postComments"
            )

        pipeline = [
            {"$match": {"title": "first", "comments.author_name": "Ada"}},
            {
                "$project": {
                    "title": 1,
                    "comment": "$comments",
                    "upper_title": {"$toUpper": "$title"},
                    "author_names": {
                        "$map": {
                            "input": "$comments",
                            "as": "comment",
                            "in": {"$toUpper": "$$comment.author_name"},
                        }
                    },
                    "active": {
                        "$filter": {
                            "input": "$comments",
                            "as": "comment",
                            "cond": {"$eq": ["$$comment.author_name", "$title"]},
                        }
                    },
                    "root_title": "$$ROOT.title",
                }
            },
            {"$match": {"upper_title": "FIRST"}},
            {"$unwind": "$comment"},
            {"$match": {"comment.author_name": "Ada"}},
            {"$rename": {"comment": "renamedComment"}},
            {"$match": {"renamedComment.author_name": "Ada"}},
        ]
        original_pipeline = copy.deepcopy(pipeline)

        converted = transform_pipeline(AggPost, pipeline)

        assert converted == [
            {
                "$match": {
                    "postTitle": "first",
                    "postComments.authorName": "Ada",
                }
            },
            {
                "$project": {
                    "postTitle": True,
                    "comment": "$postComments",
                    "upper_title": {"$toUpper": "$postTitle"},
                    "author_names": {
                        "$map": {
                            "input": "$postComments",
                            "as": "comment",
                            "in": {"$toUpper": "$$comment.authorName"},
                        }
                    },
                    "active": {
                        "$filter": {
                            "input": "$postComments",
                            "as": "comment",
                            "cond": {"$eq": ["$$comment.authorName", "$postTitle"]},
                        }
                    },
                    "root_title": "$$ROOT.postTitle",
                }
            },
            {"$match": {"upper_title": "FIRST"}},
            {"$unwind": "$comment"},
            {"$match": {"comment.authorName": "Ada"}},
            {"$rename": {"comment": "renamedComment"}},
            {"$match": {"renamedComment.authorName": "Ada"}},
        ]
        assert pipeline == original_pipeline
        encode({"pipeline": converted})

    def test_requires_raw_for_unanalyzable_stage(self):
        class RawDoc(Document):
            name = StringField(db_field="realName")

        with pytest.raises(InvalidQueryError, match=r"Use \$raw.*\$lookup"):
            transform_pipeline(
                RawDoc,
                [
                    {
                        "$lookup": {
                            "from": "other",
                            "localField": "name",
                            "foreignField": "name",
                            "as": "others",
                        }
                    }
                ],
            )

        converted = transform_pipeline(
            RawDoc,
            [
                {
                    "$raw": {
                        "$lookup": {
                            "from": "other",
                            "localField": "name",
                            "foreignField": "name",
                            "as": "others",
                        }
                    }
                },
                {"$match": {"others.name": "Ada"}},
            ],
        )
        assert converted[0]["$lookup"]["localField"] == "name"
        assert converted[1] == {"$match": {"others.name": "Ada"}}

    def test_queryset_aggregate_model_fields_does_not_mutate_pipeline(self):
        class NonMutatingDoc(Document):
            display_name = StringField(db_field="displayName")

        pipeline = [{"$match": {"display_name": "Ada"}}]
        original_pipeline = copy.deepcopy(pipeline)

        with patch.object(NonMutatingDoc, "_collection") as collection:
            NonMutatingDoc.objects.aggregate(pipeline, model_fields=True)

        assert pipeline == original_pipeline
        assert collection.aggregate.call_args.args[0] == [
            {"$match": {"displayName": "Ada"}}
        ]

    def test_queryset_aggregate_default_passes_pipeline_unchanged(self):
        class DefaultDoc(Document):
            display_name = StringField(db_field="displayName")

        pipeline = [{"$match": {"display_name": "Ada"}}]

        with patch.object(DefaultDoc, "_collection") as collection:
            DefaultDoc.objects.aggregate(pipeline)

        assert collection.aggregate.call_args.args[0] == pipeline


class TestQuerysetAggregate(MongoDBTestCase):
    def test_read_preference_aggregation_framework(self):
        class Bar(Document):
            txt = StringField()

            meta = {"indexes": ["txt"]}

        # Aggregates with read_preference
        pipeline = []
        bars = Bar.objects.read_preference(
            ReadPreference.SECONDARY_PREFERRED
        ).aggregate(pipeline)
        if hasattr(bars, "_CommandCursor__collection"):
            read_pref = bars._CommandCursor__collection.read_preference
        else:  # pymongo >= 4.9
            read_pref = bars._collection.read_preference
        assert read_pref == ReadPreference.SECONDARY_PREFERRED

    def test_queryset_aggregation_framework(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects(age__lte=22).aggregate(pipeline)

        assert list(data) == [
            {"_id": p1.pk, "name": "ISABELLA LUANNA"},
            {"_id": p2.pk, "name": "WILSON JUNIOR"},
        ]

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects(age__lte=22).order_by("-name").aggregate(pipeline)

        assert list(data) == [
            {"_id": p2.pk, "name": "WILSON JUNIOR"},
            {"_id": p1.pk, "name": "ISABELLA LUANNA"},
        ]

        pipeline = [
            {"$group": {"_id": None, "total": {"$sum": 1}, "avg": {"$avg": "$age"}}}
        ]
        data = (
            Person.objects(age__gte=17, age__lte=40)
            .order_by("-age")
            .aggregate(pipeline)
        )
        assert list(data) == [{"_id": None, "avg": 29, "total": 2}]

        pipeline = [{"$match": {"name": "Isabella Luanna"}}]
        data = Person.objects().aggregate(pipeline)
        assert list(data) == [{"_id": p1.pk, "age": 16, "name": "Isabella Luanna"}]

    def test_queryset_aggregation_with_skip(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects.skip(1).aggregate(pipeline)

        assert list(data) == [
            {"_id": p2.pk, "name": "WILSON JUNIOR"},
            {"_id": p3.pk, "name": "SANDRA MARA"},
        ]

    def test_aggregation_propagates_hint_collation_and_comment(self):
        """Make sure adding a hint/comment/collation to the query gets added to the query"""
        mongo_ver = get_mongodb_version()

        base = {"locale": "en", "strength": 2}
        index_name = "name_1"

        class AggPerson(Document):
            name = StringField()
            meta = {
                "indexes": [{"fields": ["name"], "name": index_name, "collation": base}]
            }

        AggPerson.drop_collection()
        _ = AggPerson.objects.first()

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        comment = "test_comment"

        with db_ops_tracker() as q:
            _ = list(AggPerson.objects.comment(comment).aggregate(pipeline))
            query_op = q.db.system.profile.find({"ns": "mongoenginetest.agg_person"})[0]
            CMD_QUERY_KEY = "command" if mongo_ver >= MONGODB_36 else "query"
            assert "hint" not in query_op[CMD_QUERY_KEY]
            assert query_op[CMD_QUERY_KEY]["comment"] == comment
            assert "collation" not in query_op[CMD_QUERY_KEY]

        with db_ops_tracker() as q:
            _ = list(AggPerson.objects.hint(index_name).aggregate(pipeline))
            query_op = q.db.system.profile.find({"ns": "mongoenginetest.agg_person"})[0]
            CMD_QUERY_KEY = "command" if mongo_ver >= MONGODB_36 else "query"
            assert query_op[CMD_QUERY_KEY]["hint"] == "name_1"
            assert "comment" not in query_op[CMD_QUERY_KEY]
            assert "collation" not in query_op[CMD_QUERY_KEY]

        with db_ops_tracker() as q:
            _ = list(AggPerson.objects.collation(base).aggregate(pipeline))
            query_op = q.db.system.profile.find({"ns": "mongoenginetest.agg_person"})[0]
            CMD_QUERY_KEY = "command" if mongo_ver >= MONGODB_36 else "query"
            assert "hint" not in query_op[CMD_QUERY_KEY]
            assert "comment" not in query_op[CMD_QUERY_KEY]
            assert query_op[CMD_QUERY_KEY]["collation"] == base

    def test_queryset_aggregation_with_limit(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects.limit(1).aggregate(pipeline)

        assert list(data) == [{"_id": p1.pk, "name": "ISABELLA LUANNA"}]

    def test_queryset_aggregation_with_sort(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects.order_by("name").aggregate(pipeline)

        assert list(data) == [
            {"_id": p1.pk, "name": "ISABELLA LUANNA"},
            {"_id": p3.pk, "name": "SANDRA MARA"},
            {"_id": p2.pk, "name": "WILSON JUNIOR"},
        ]

    def test_queryset_aggregation_with_skip_with_limit(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = list(Person.objects.skip(1).limit(1).aggregate(pipeline))

        assert list(data) == [{"_id": p2.pk, "name": "WILSON JUNIOR"}]

        # Make sure limit/skip chaining order has no impact
        data2 = Person.objects.limit(1).skip(1).aggregate(pipeline)

        assert data == list(data2)

    def test_queryset_aggregation_with_sort_with_limit(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects.order_by("name").limit(2).aggregate(pipeline)

        assert list(data) == [
            {"_id": p1.pk, "name": "ISABELLA LUANNA"},
            {"_id": p3.pk, "name": "SANDRA MARA"},
        ]

        # Verify adding limit/skip steps works as expected
        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}, {"$limit": 1}]
        data = Person.objects.order_by("name").limit(2).aggregate(pipeline)

        assert list(data) == [{"_id": p1.pk, "name": "ISABELLA LUANNA"}]

        pipeline = [
            {"$project": {"name": {"$toUpper": "$name"}}},
            {"$skip": 1},
            {"$limit": 1},
        ]
        data = Person.objects.order_by("name").limit(2).aggregate(pipeline)

        assert list(data) == [{"_id": p3.pk, "name": "SANDRA MARA"}]

    def test_queryset_aggregation_with_sort_with_skip(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects.order_by("name").skip(2).aggregate(pipeline)

        assert list(data) == [{"_id": p2.pk, "name": "WILSON JUNIOR"}]

    def test_queryset_aggregation_with_sort_with_skip_with_limit(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects.order_by("name").skip(1).limit(1).aggregate(pipeline)

        assert list(data) == [{"_id": p3.pk, "name": "SANDRA MARA"}]

    def test_queryset_aggregation_old_interface_not_working(self):
        class Person(Document):
            name = StringField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna")
        p2 = Person(name="Wilson Junior")
        p3 = Person(name="Sandra Mara")
        Person.objects.insert([p1, p2, p3])

        _1_step_pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]

        # Make sure old interface raises an error as we changed it >= 1.0
        with pytest.raises(TypeError, match="pipeline must be a list/tuple"):
            Person.objects.order_by("name").limit(2).aggregate(*_1_step_pipeline)

        _2_step_pipeline = [
            {"$project": {"name": {"$toUpper": "$name"}}},
            {"$limit": 1},
        ]
        with pytest.raises(
            TypeError, match="takes 2 positional arguments but 3 were given"
        ):
            Person.objects.order_by("name").limit(2).aggregate(*_2_step_pipeline)

    def test_queryset_aggregation_geonear_aggregation_on_pointfield(self):
        """test ensures that $geonear can be used as a 1-stage pipeline and that
        MongoEngine does not interfer with such pipeline (#2473)
        """

        class Aggr(Document):
            name = StringField()
            c = PointField()

        Aggr.drop_collection()

        agg1 = Aggr(name="X", c=[10.634584, 35.8245029]).save()
        agg2 = Aggr(name="Y", c=[10.634584, 35.8245029]).save()

        pipeline = [
            {
                "$geoNear": {
                    "near": {"type": "Point", "coordinates": [10.634584, 35.8245029]},
                    "distanceField": "c",
                    "spherical": True,
                }
            }
        ]
        assert list(Aggr.objects.aggregate(pipeline)) == [
            {"_id": agg1.id, "c": 0.0, "name": "X"},
            {"_id": agg2.id, "c": 0.0, "name": "Y"},
        ]

    def test_queryset_aggregation_none(self):
        class Person(Document):
            name = StringField()
            age = IntField()

        Person.drop_collection()

        p1 = Person(name="Isabella Luanna", age=16)
        p2 = Person(name="Wilson Junior", age=21)
        p3 = Person(name="Sandra Mara", age=37)
        Person.objects.insert([p1, p2, p3])

        pipeline = [{"$project": {"name": {"$toUpper": "$name"}}}]
        data = Person.objects().none().order_by("name").aggregate(pipeline)

        assert list(data) == []

    def test_aggregate_geo_near_used_as_initial_step_before_cls_implicit_step(self):
        class BaseClass(Document):
            meta = {"allow_inheritance": True}

        class Aggr(BaseClass):
            name = StringField()
            c = PointField()

        BaseClass.drop_collection()

        x = Aggr(name="X", c=[10.634584, 35.8245029]).save()
        y = Aggr(name="Y", c=[10.634584, 35.8245029]).save()

        pipeline = [
            {
                "$geoNear": {
                    "near": {"type": "Point", "coordinates": [10.634584, 35.8245029]},
                    "distanceField": "c",
                    "spherical": True,
                }
            }
        ]
        res = list(Aggr.objects.aggregate(pipeline))
        assert res == [
            {"_cls": "BaseClass.Aggr", "_id": x.id, "c": 0.0, "name": "X"},
            {"_cls": "BaseClass.Aggr", "_id": y.id, "c": 0.0, "name": "Y"},
        ]

    def test_aggregate_collstats_used_as_initial_step_before_cls_implicit_step(self):
        class SomeDoc(Document):
            name = StringField()

        SomeDoc.drop_collection()

        SomeDoc(name="X").save()
        SomeDoc(name="Y").save()

        pipeline = [{"$collStats": {"count": {}}}]
        res = list(SomeDoc.objects.aggregate(pipeline))
        assert len(res) == 1
        assert res[0]["count"] == 2

    def test_aggregate__search_stage__precedes_implicit_match(self):
        class SearchableDoc(Document):
            last_name = StringField()

        for search_stage in ("$search", "$searchMeta", "$vectorSearch"):
            with self.subTest(search_stage=search_stage):
                search_step = {search_stage: {}}

                # End-to-end execution requires MongoDB Search (mongot), so mock
                # the collection and verify the pipeline passed to PyMongo.
                with patch.object(SearchableDoc, "_collection") as collection:
                    SearchableDoc.objects(last_name="bar").aggregate([search_step])

                assert collection.aggregate.call_args.args[0] == [
                    search_step,
                    {"$match": {"last_name": "bar"}},
                ]

    def test_queryset_aggregate_model_fields_end_to_end(self):
        class AggComment(EmbeddedDocument):
            author_name = StringField(db_field="authorName")

        class AggPost(Document):
            title = StringField(db_field="postTitle")
            comments = ListField(
                EmbeddedDocumentField(AggComment), db_field="postComments"
            )

        AggPost.drop_collection()
        AggPost(
            title="first",
            comments=[
                AggComment(author_name="Ada"),
                AggComment(author_name="Grace"),
            ],
        ).save()
        AggPost(title="second", comments=[AggComment(author_name="Katherine")]).save()

        pipeline = [
            {"$match": {"title": "first"}},
            {
                "$project": {
                    "title": 1,
                    "comment": "$comments",
                    "author_names": {
                        "$map": {
                            "input": "$comments",
                            "as": "comment",
                            "in": "$$comment.author_name",
                        }
                    },
                    "is_ada": {
                        "$filter": {
                            "input": "$comments",
                            "as": "comment",
                            "cond": {"$eq": ["$$comment.author_name", "Ada"]},
                        }
                    },
                }
            },
            {"$unwind": "$comment"},
            {"$match": {"comment.author_name": "Ada"}},
            {"$project": {"_id": 0, "title": 1, "author_names": 1}},
        ]

        result = list(AggPost.objects.aggregate(pipeline, model_fields=True))

        assert len(result) == 1
        assert result[0]["postTitle"] == "first"
        assert result[0]["author_names"] == ["Ada", "Grace"]
