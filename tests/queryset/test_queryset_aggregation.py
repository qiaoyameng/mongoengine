import copy
import unittest
from unittest.mock import patch

import pytest
from pymongo.read_preferences import ReadPreference

try:
    import mongomock
except ImportError:
    mongomock = None

from mongoengine import (
    Document,
    EmbeddedDocument,
    EmbeddedDocumentField,
    IntField,
    ListField,
    PointField,
    StringField,
    connect,
)
from mongoengine.connection import disconnect_all
from mongoengine.errors import InvalidQueryError
from mongoengine.mongodb_support import (
    MONGODB_36,
    get_mongodb_version,
)
from mongoengine.queryset.aggregate import raw, translate_pipeline
from tests.utils import MongoDBTestCase, db_ops_tracker


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


class TranslatedOrderItem(EmbeddedDocument):
    sku = StringField(db_field="s")
    price = IntField(db_field="p")


class TranslatedOrder(Document):
    name = StringField(db_field="n")
    total = IntField(db_field="t")
    items = ListField(EmbeddedDocumentField(TranslatedOrderItem), db_field="itms")

    meta = {"collection": "translated_order", "allow_inheritance": False}


class TestPipelineFieldTranslation(unittest.TestCase):
    """Unit tests for mongoengine.queryset.aggregate.translate_pipeline"""

    def test_match_and_sort_use_db_field_names(self):
        pipeline = [
            {"$match": {"name": "foo", "total": {"$gte": 3}}},
            {"$sort": {"total": -1}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$match": {"n": "foo", "t": {"$gte": 3}}},
            {"$sort": {"t": -1}},
        ]

    def test_match_embedded_paths_and_elem_match(self):
        pipeline = [
            {"$match": {"items": {"$elemMatch": {"sku": "A", "price": {"$gt": 5}}}}},
            {"$match": {"items.price": 7}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$match": {"itms": {"$elemMatch": {"s": "A", "p": {"$gt": 5}}}}},
            {"$match": {"itms.p": 7}},
        ]

    def test_match_expr_and_logical_operators(self):
        pipeline = [
            {
                "$match": {
                    "$or": [
                        {"name": "foo"},
                        {"$expr": {"$gt": ["$total", 10]}},
                    ]
                }
            }
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$match": {"$or": [{"n": "foo"}, {"$expr": {"$gt": ["$t", 10]}}]}}
        ]

    def test_project_tracks_computed_fields(self):
        pipeline = [
            {"$project": {"upper": {"$toUpper": "$name"}, "total": 1}},
            {"$match": {"upper": "FOO"}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$project": {"upper": {"$toUpper": "$n"}, "t": 1}},
            # "upper" is a computed field: it must not be renamed
            {"$match": {"upper": "FOO"}},
        ]

    def test_project_renamed_field_reference(self):
        pipeline = [
            {"$project": {"renamed": "$name"}},
            {"$sort": {"renamed": 1}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$project": {"renamed": "$n"}},
            {"$sort": {"renamed": 1}},
        ]

    def test_projected_away_field_cannot_be_referenced(self):
        pipeline = [{"$project": {"name": 1}}, {"$match": {"total": 3}}]
        with pytest.raises(InvalidQueryError):
            translate_pipeline(TranslatedOrder, pipeline)

    def test_project_nested_inclusion_and_exclusion(self):
        pipeline = [{"$project": {"items": {"sku": 1}, "name": 1, "_id": 0}}]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$project": {"itms": {"s": 1}, "n": 1, "_id": 0}}
        ]

    def test_unwind_translates_path(self):
        pipeline = [{"$unwind": "$items"}, {"$match": {"items.sku": "A"}}]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$unwind": "$itms"},
            {"$match": {"itms.s": "A"}},
        ]

    def test_unwind_options_form(self):
        pipeline = [
            {"$unwind": {"path": "$items", "includeArrayIndex": "idx"}},
            {"$sort": {"idx": 1}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$unwind": {"path": "$itms", "includeArrayIndex": "idx"}},
            {"$sort": {"idx": 1}},
        ]

    def test_map_filter_and_root_variable_scope(self):
        pipeline = [
            {
                "$project": {
                    "prices": {
                        "$map": {"input": "$items", "as": "it", "in": "$$it.price"}
                    },
                    "cheap": {
                        "$filter": {
                            "input": "$items",
                            "as": "it",
                            "cond": {"$lt": ["$$it.price", "$$ROOT.total"]},
                        }
                    },
                }
            }
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {
                "$project": {
                    "prices": {"$map": {"input": "$itms", "as": "it", "in": "$$it.p"}},
                    "cheap": {
                        "$filter": {
                            "input": "$itms",
                            "as": "it",
                            "cond": {"$lt": ["$$it.p", "$$ROOT.t"]},
                        }
                    },
                }
            }
        ]

    def test_let_binds_opaque_variables(self):
        pipeline = [
            {
                "$project": {
                    "doubled": {
                        "$let": {
                            "vars": {"base": "$total"},
                            "in": {"$multiply": ["$$base", 2]},
                        }
                    }
                }
            }
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {
                "$project": {
                    "doubled": {
                        "$let": {
                            "vars": {"base": "$t"},
                            "in": {"$multiply": ["$$base", 2]},
                        }
                    }
                }
            }
        ]

    def test_group_and_downstream_computed_fields(self):
        pipeline = [
            {"$group": {"_id": "$name", "total_sum": {"$sum": "$total"}}},
            {"$sort": {"total_sum": -1}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$group": {"_id": "$n", "total_sum": {"$sum": "$t"}}},
            {"$sort": {"total_sum": -1}},
        ]

    def test_add_fields_then_reference(self):
        pipeline = [
            {"$addFields": {"grand": {"$add": ["$total", 1]}}},
            {"$match": {"grand": {"$gt": 3}}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$addFields": {"grand": {"$add": ["$t", 1]}}},
            {"$match": {"grand": {"$gt": 3}}},
        ]

    def test_unset_removes_fields_from_scope(self):
        pipeline = [{"$unset": "total"}, {"$match": {"total": 1}}]
        with pytest.raises(InvalidQueryError):
            translate_pipeline(TranslatedOrder, pipeline)
        pipeline = [{"$unset": ["total", "items.price"]}]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$unset": ["t", "itms.p"]}
        ]

    def test_replace_root_with_known_subdocument(self):
        pipeline = [
            {"$unwind": "$items"},
            {"$replaceRoot": {"newRoot": "$items"}},
            {"$match": {"sku": "A"}},
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$unwind": "$itms"},
            {"$replaceRoot": {"newRoot": "$itms"}},
            {"$match": {"s": "A"}},
        ]

    def test_unknown_field_raises(self):
        with pytest.raises(InvalidQueryError):
            translate_pipeline(TranslatedOrder, [{"$match": {"does_not_exist": 1}}])

    def test_unknown_stage_requires_raw(self):
        lookup = {
            "$lookup": {
                "from": "other",
                "localField": "n",
                "foreignField": "x",
                "as": "joined",
            }
        }
        with pytest.raises(InvalidQueryError, match="raw"):
            translate_pipeline(TranslatedOrder, [lookup])
        pipeline = [{"$match": {"name": "x"}}, raw(lookup)]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {"$match": {"n": "x"}},
            lookup,
        ]

    def test_raw_rejects_non_dict(self):
        with pytest.raises(TypeError):
            raw("$match")

    def test_input_pipeline_is_not_mutated(self):
        pipeline = [
            {"$match": {"name": "foo"}},
            {"$project": {"upper": {"$toUpper": "$name"}}},
            raw({"$lookup": {"from": "c", "as": "j", "pipeline": []}}),
        ]
        snapshot = copy.deepcopy(pipeline)
        translate_pipeline(TranslatedOrder, pipeline)
        assert pipeline == snapshot

    def test_facet_translates_sub_pipelines(self):
        pipeline = [
            {
                "$facet": {
                    "names": [{"$sortByCount": "$name"}],
                    "totals": [{"$group": {"_id": None, "s": {"$sum": "$total"}}}],
                }
            }
        ]
        assert translate_pipeline(TranslatedOrder, pipeline) == [
            {
                "$facet": {
                    "names": [{"$sortByCount": "$n"}],
                    "totals": [{"$group": {"_id": None, "s": {"$sum": "$t"}}}],
                }
            }
        ]


class TestQuerysetAggregateTranslateFields(unittest.TestCase):
    """Integration of translate_fields=True with QuerySet.aggregate"""

    def test_translated_pipeline_is_passed_to_pymongo(self):
        pipeline = [
            {"$unwind": "$items"},
            {"$match": {"items.price": {"$gte": 10}}},
            {"$project": {"sku": "$items.sku", "order_total": "$$ROOT.total"}},
        ]
        with patch.object(TranslatedOrder, "_collection") as collection:
            TranslatedOrder.objects(total__gte=1).aggregate(
                pipeline, translate_fields=True
            )

        assert collection.aggregate.call_args.args[0] == [
            {"$match": {"t": {"$gte": 1}}},
            {"$unwind": "$itms"},
            {"$match": {"itms.p": {"$gte": 10}}},
            {"$project": {"sku": "$itms.s", "order_total": "$$ROOT.t"}},
        ]

    def test_translate_fields_does_not_mutate_user_pipeline(self):
        pipeline = [{"$project": {"upper": {"$toUpper": "$name"}}}]
        snapshot = copy.deepcopy(pipeline)
        with patch.object(TranslatedOrder, "_collection"):
            TranslatedOrder.objects.aggregate(pipeline, translate_fields=True)
        assert pipeline == snapshot

    def test_translate_fields_defaults_to_false(self):
        pipeline = [{"$match": {"n": "foo"}}]
        with patch.object(TranslatedOrder, "_collection") as collection:
            TranslatedOrder.objects.aggregate(pipeline)
        assert collection.aggregate.call_args.args[0] == [{"$match": {"n": "foo"}}]


@unittest.skipIf(mongomock is None, "mongomock is not installed")
class TestQuerysetAggregateTranslateFieldsMongomock(unittest.TestCase):
    """End-to-end execution of translated pipelines against mongomock"""

    @classmethod
    def setUpClass(cls):
        disconnect_all()
        cls._connection = connect(
            db="mongoenginetest", mongo_client_class=mongomock.MongoClient
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        TranslatedOrder.drop_collection()
        TranslatedOrder(
            name="a",
            total=5,
            items=[
                TranslatedOrderItem(sku="x", price=3),
                TranslatedOrderItem(sku="y", price=30),
            ],
        ).save()
        TranslatedOrder(
            name="b", total=50, items=[TranslatedOrderItem(sku="z", price=12)]
        ).save()

    def test_unwind_match_project_with_root_variable(self):
        pipeline = [
            {"$unwind": "$items"},
            {"$match": {"items.price": {"$gte": 10}}},
            {
                "$project": {
                    "_id": 0,
                    "sku": "$items.sku",
                    "order_total": "$$ROOT.total",
                }
            },
            {"$sort": {"sku": 1}},
        ]
        data = list(TranslatedOrder.objects.aggregate(pipeline, translate_fields=True))
        assert data == [
            {"sku": "y", "order_total": 5},
            {"sku": "z", "order_total": 50},
        ]

    def test_project_with_filter_variable(self):
        pipeline = [
            {
                "$project": {
                    "_id": 0,
                    "name": 1,
                    "expensive": {
                        "$filter": {
                            "input": "$items",
                            "as": "it",
                            "cond": {"$gte": ["$$it.price", 10]},
                        }
                    },
                }
            },
            {"$sort": {"name": 1}},
        ]
        data = list(TranslatedOrder.objects.aggregate(pipeline, translate_fields=True))
        assert data == [
            {"n": "a", "expensive": [{"s": "y", "p": 30}]},
            {"n": "b", "expensive": [{"s": "z", "p": 12}]},
        ]

    def test_group_with_queryset_filter(self):
        pipeline = [{"$group": {"_id": None, "s": {"$sum": "$total"}}}]
        data = list(
            TranslatedOrder.objects(total__gte=10).aggregate(
                pipeline, translate_fields=True
            )
        )
        assert data == [{"_id": None, "s": 50}]
