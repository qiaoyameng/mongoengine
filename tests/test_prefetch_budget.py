"""Tests for the bounded prefetch engine (``mongoengine.prefetch``).

These tests run against mongomock and, most importantly, wrap the PyMongo
collections in :class:`CountingCollectionStub` so that the *exact* number of
queries issued by a prefetch run can be asserted (a raw mongomock client
makes that difficult to observe).
"""

import json
import unittest

import pytest
from bson import json_util

import mongoengine
from mongoengine import (
    Document,
    EmbeddedDocument,
    GenericReferenceField,
    LazyReferenceField,
    ListField,
    ReferenceField,
    StringField,
    UnresolvedReference,
    connect,
    disconnect_all,
    prefetch_related,
)
from mongoengine.base.datastructures import LazyReference
from mongoengine.errors import PrefetchBudgetExhausted
from mongoengine.prefetch import PrefetchBudget, PrefetchReport, last_report

try:
    import mongomock

    MONGOMOCK_INSTALLED = True
except ImportError:  # pragma: no cover
    MONGOMOCK_INSTALLED = False

require_mongomock = pytest.mark.skipif(
    not MONGOMOCK_INSTALLED, reason="you need mongomock installed"
)


class CountingCursorStub:
    """Cursor wrapper counting how many documents get materialised.

    It is a fully transparent proxy so PyMongo cursor features (slicing,
    cloning, context-manager use ...) keep working unchanged.
    """

    def __init__(self, real_cursor, stub):
        object.__setattr__(self, "_real_cursor", real_cursor)
        object.__setattr__(self, "_stub", stub)

    def __iter__(self):
        return self

    def __next__(self):
        doc = next(self._real_cursor)
        self._stub.documents_returned += 1
        return doc

    def __getitem__(self, index):
        result = self._real_cursor[index]
        if isinstance(result, type(self._real_cursor)):
            return CountingCursorStub(result, self._stub)
        return result

    def __len__(self):
        return len(self._real_cursor)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def __class__(self):
        return type(self._real_cursor)

    def __getattr__(self, name):
        return getattr(self._real_cursor, name)


class CountingCollectionStub:
    """Wraps a real/mongomock collection and counts the queries it serves.

    Only read operations used by prefetching (``find`` / ``find_one`` ...) are
    counted; everything else is transparently delegated.
    """

    def __init__(self, real_collection):
        self._real_collection = real_collection
        self.name = real_collection.name
        self.find_calls = 0
        self.documents_returned = 0
        self.filters_seen = []

    def find(self, *args, **kwargs):
        self.find_calls += 1
        self.filters_seen.append(args[0] if args else None)
        cursor = self._real_collection.find(*args, **kwargs)
        return CountingCursorStub(cursor, self)

    def __getattr__(self, name):
        return getattr(self._real_collection, name)

    def reset(self):
        self.find_calls = 0
        self.documents_returned = 0
        self.filters_seen = []


class CountingRegistry:
    """Installs counting stubs on a set of document classes."""

    def __init__(self, *classes):
        self.classes = classes
        self.stubs = {}

    def install(self):
        for cls in self.classes:
            real = cls._get_collection()
            stub = CountingCollectionStub(real)
            cls._collection = stub
            self.stubs[cls._get_collection_name()] = stub
        return self

    def remove(self):
        for cls in self.classes:
            cls._collection = None
        self.stubs.clear()

    @property
    def find_calls(self):
        return sum(stub.find_calls for stub in self.stubs.values())

    @property
    def documents_returned(self):
        return sum(stub.documents_returned for stub in self.stubs.values())

    def reset(self):
        for stub in self.stubs.values():
            stub.reset()


@require_mongomock
class PrefetchBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchbudgettest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        class User(Document):
            name = StringField()
            meta = {"allow_inheritance": False}

        class Post(Document):
            title = StringField()
            author = ReferenceField(User)
            meta = {"allow_inheritance": False}

        self.User = User
        self.Post = Post
        User.drop_collection()
        Post.drop_collection()

    def test_budget_validation(self):
        PrefetchBudget(max_depth=0)
        PrefetchBudget(max_depth=None, max_objects=0, max_queries_per_round=1)
        with self.assertRaises(ValueError):
            PrefetchBudget(max_depth=-1)
        with self.assertRaises(ValueError):
            PrefetchBudget(max_queries_per_round=0)
        with self.assertRaises(TypeError):
            PrefetchBudget(max_depth=1.5)
        with self.assertRaises(TypeError):
            PrefetchBudget(max_depth=True)

    def test_direct_references_one_batched_query(self):
        User, Post = self.User, self.Post
        users = [User(name=f"u{i}").save() for i in range(5)]
        for i in range(4):
            Post(title=f"p{i}", author=users[i]).save()

        registry = CountingRegistry(User, Post).install()
        try:
            posts = list(Post.objects())
            registry.reset()
            report = prefetch_related(posts, max_depth=1)
            self.assertEqual(report.query_count, 1)
            self.assertEqual(registry.find_calls, 1)
            self.assertEqual(report.objects_fetched, 4)
            self.assertFalse(report.incomplete)
            self.assertEqual([p.author.name for p in posts], ["u0", "u1", "u2", "u3"])
        finally:
            registry.remove()

    def test_shared_reference_counts_unique_object_once(self):
        """Same object referenced from many paths: 1 query, 1 unique object,
        and every slot points to the same in-memory instance."""
        User, Post = self.User, self.Post
        shared = User(name="shared").save()
        posts = [Post(title=f"p{i}", author=shared).save() for i in range(5)]

        registry = CountingRegistry(User, Post).install()
        try:
            roots = list(Post.objects())
            registry.reset()
            report = prefetch_related(roots, max_depth=1)
            self.assertEqual(report.query_count, 1)
            self.assertEqual(report.objects_fetched, 1)
            self.assertEqual(len({id(p.author) for p in roots}), 1)
            self.assertTrue(all(p.author is shared or p.author.name == "shared" for p in roots))
        finally:
            registry.remove()

    def test_object_budget_caps_unique_objects(self):
        User, Post = self.User, self.Post
        users = [User(name=f"u{i}").save() for i in range(4)]
        posts = [Post(title=f"p{i}", author=users[i]).save() for i in range(4)]

        registry = CountingRegistry(User, Post).install()
        try:
            roots = list(Post.objects())
            registry.reset()
            report = prefetch_related(roots, max_depth=1, max_objects=2)
            self.assertEqual(report.objects_fetched, 2)
            self.assertTrue(report.incomplete)
            resolved = [p for p in roots if isinstance(p._data["author"], User)]
            unresolved = [
                p for p in roots if isinstance(p._data["author"], UnresolvedReference)
            ]
            self.assertEqual(len(resolved), 2)
            self.assertEqual(len(unresolved), 2)
            for post in unresolved:
                marker = post._data["author"]
                self.assertEqual(marker.reason, "objects")
                self.assertTrue(marker.path.endswith(".author"))
                # strictly lazy
                with self.assertRaises(AttributeError):
                    marker.some_field
                with self.assertRaises(PrefetchBudgetExhausted):
                    marker.fetch()
                # printing / serialisation never queries
                str(marker)
                repr(marker)
                json.loads(post.to_json(json_options=json_util.DEFAULT_JSON_OPTIONS))
            self.assertEqual(registry.find_calls, report.query_count)
        finally:
            registry.remove()

    def test_queries_per_round_cap_returns_report_with_paths(self):
        User, Post = self.User, self.Post
        users = [User(name=f"u{i}").save() for i in range(3)]
        [Post(title=f"p{i}", author=users[i]).save() for i in range(3)]

        # Both references live in the same collection, so the per-round cap is
        # exercised through a chain on the same collection below; here we use
        # a tiny object cap to split a group over rounds and assert no round
        # exceeds the configured query ceiling.
        registry = CountingRegistry(User, Post).install()
        try:
            roots = list(Post.objects())
            registry.reset()
            report = prefetch_related(
                roots, max_depth=1, max_objects=1, max_queries_per_round=1
            )
            self.assertTrue(all(n <= 1 for n in report.queries_by_round))
            self.assertEqual(report.query_count, registry.find_calls)
        finally:
            registry.remove()

    def test_depth_budget_marks_boundary(self):
        User, Post = self.User, self.Post
        u = User(name="u").save()
        p = Post(title="p", author=u).save()

        registry = CountingRegistry(User, Post).install()
        try:
            root = Post.objects.first()
            report = prefetch_related(root, max_depth=0)
            self.assertEqual(report.query_count, 0)
            self.assertIsInstance(root._data["author"], UnresolvedReference)
            self.assertEqual(root._data["author"].reason, "depth")
            self.assertIn("root[0].author", report.unresolved)

            root = Post.objects.first()
            report = prefetch_related(root, max_depth=1)
            self.assertIsInstance(root._data["author"], User)
            self.assertFalse(report.incomplete)
        finally:
            registry.remove()

    def test_report_describes_unresolved_paths(self):
        User, Post = self.User, self.Post
        users = [User(name=f"u{i}").save() for i in range(2)]
        [Post(title="p", author=users[i]).save() for i in range(2)]
        roots = list(Post.objects())
        report = prefetch_related(roots, max_depth=1, max_objects=1)
        text = report.describe_unresolved()
        self.assertIn("author", text)
        self.assertTrue(any(p.endswith(".author") for p in report.unresolved_paths))


@require_mongomock
class PrefetchGraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchgraphtest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        class Node(Document):
            name = StringField()
            next = ReferenceField("self")
            friends = ListField(ReferenceField("self"))
            meta = {"allow_inheritance": False}

        self.Node = Node
        Node.drop_collection()

    def _make_cycle(self, n=4, loop_back=True):
        Node = self.Node
        nodes = [Node(name=f"n{i}") for i in range(n)]
        for node in nodes:
            node.save()
        for i in range(n - 1):
            nodes[i].next = nodes[i + 1]
            nodes[i].save()
        if loop_back:
            nodes[-1].next = nodes[0]
            nodes[-1].save()
        return nodes

    def test_cycle_traversed_once_per_unique_object(self):
        self._make_cycle(n=4)
        Node = self.Node
        registry = CountingRegistry(Node).install()
        try:
            root = Node.objects(name="n0").first()
            registry.reset()
            report = prefetch_related(root, max_depth=None)
            self.assertFalse(report.incomplete, report.describe_unresolved())
            # 4 nodes total, 1 root already in memory -> 3 fetched, 3 rounds
            self.assertEqual(report.objects_fetched, 3)
            self.assertEqual(report.query_count, 3)
            self.assertEqual(report.rounds, 3)
            # closes the cycle: n0 -> n1 -> n2 -> n3 -> n0
            self.assertIs(root.next.next.next.next, root)
            self.assertEqual(registry.find_calls, 3)
        finally:
            registry.remove()

    def test_cycle_with_depth_limit(self):
        self._make_cycle(n=4)
        Node = self.Node
        root = Node.objects(name="n0").first()
        report = prefetch_related(root, max_depth=2)
        self.assertTrue(report.incomplete)
        self.assertEqual(report.objects_fetched, 2)
        marker = root.next.next._data["next"]
        self.assertIsInstance(marker, UnresolvedReference)
        self.assertEqual(marker.reason, "depth")
        self.assertIn("next.next.next", marker.path)

    def test_diamond_shared_node(self):
        Node = self.Node
        a = Node(name="a"); b = Node(name="b"); c = Node(name="c")
        d = Node(name="d")
        for node in (a, b, c, d):
            node.save()
        a.friends = [b, c]; a.save()
        b.next = d; b.save()
        c.next = d; c.save()
        root = Node.objects(name="a").first()
        report = prefetch_related(root, max_depth=2)
        self.assertFalse(report.incomplete, report.describe_unresolved())
        # b,c at depth1, d reached via both b and c at depth2 -> fetched once
        self.assertEqual(report.objects_fetched, 3)
        self.assertIs(root.friends[0].next, root.friends[1].next)

    def test_queries_per_round_splits_across_rounds(self):
        self._make_cycle(n=4)
        Node = self.Node
        registry = CountingRegistry(Node).install()
        try:
            root = Node.objects(name="n0").first()
            report = prefetch_related(root, max_depth=None, max_queries_per_round=1)
            self.assertTrue(all(n <= 1 for n in report.queries_by_round))
            self.assertEqual(report.query_count, 3)
        finally:
            registry.remove()


@require_mongomock
class CrossCollectionAndEmbeddingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchcrosstest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        class City(Document):
            name = StringField()
            meta = {"allow_inheritance": False}

        class Tag(Document):
            label = StringField()
            meta = {"allow_inheritance": False}

        class Address(EmbeddedDocument):
            city = ReferenceField(City)
            note = StringField()

        class Person(Document):
            name = StringField()
            hometown = ReferenceField(City)
            address = __import__("mongoengine").EmbeddedDocumentField(Address)
            tags = ListField(ReferenceField(Tag))
            favourite = GenericReferenceField()
            meta = {"allow_inheritance": False}

        self.City, self.Tag, self.Person = City, Tag, Person
        City.drop_collection()
        Tag.drop_collection()
        Person.drop_collection()

    def test_cross_collections_one_query_each(self):
        City, Tag, Person = self.City, self.Tag, self.Person
        paris = City(name="Paris").save()
        lyon = City(name="Lyon").save()
        t1 = Tag(label="a").save()
        t2 = Tag(label="b").save()
        p = Person(
            name="Bob",
            hometown=paris,
            address=Person._fields["address"].document_type(city=lyon, note="home"),
            tags=[t1, t2],
            favourite=t1,
        )
        p.save()

        registry = CountingRegistry(City, Tag, Person).install()
        try:
            root = Person.objects.first()
            registry.reset()
            report = prefetch_related(root, max_depth=1)
            self.assertFalse(report.incomplete, report.describe_unresolved())
            # one query per referenced collection (city, tag) -> 2 queries
            self.assertEqual(report.query_count, 2)
            self.assertEqual(registry.find_calls, 2)
            # unique objects: 2 cities + 2 tags = 4 (generic t1 shared)
            self.assertEqual(report.objects_fetched, 4)
            self.assertEqual(root.hometown.name, "Paris")
            self.assertEqual(root.address.city.name, "Lyon")
            self.assertEqual([t.label for t in root.tags], ["a", "b"])
            self.assertEqual(root.favourite.label, "a")
            # generic and typed edge resolve to the same instance
            self.assertIs(root.favourite, root.tags[0])
        finally:
            registry.remove()

    def test_query_round_cap_on_different_collections(self):
        City, Tag, Person = self.City, self.Tag, self.Person
        paris = City(name="Paris").save()
        t1 = Tag(label="a").save()
        Address = Person._fields["address"].document_type
        Person(
            name="Bob", hometown=paris, address=Address(city=paris), tags=[t1],
            favourite=t1,
        ).save()

        registry = CountingRegistry(City, Tag, Person).install()
        try:
            root = Person.objects.first()
            report = prefetch_related(root, max_depth=1, max_queries_per_round=1)
            self.assertTrue(all(n <= 1 for n in report.queries_by_round))
            self.assertEqual(report.query_count, 2)
            self.assertEqual(report.rounds, 2)
            self.assertEqual([1, 1], report.queries_by_round)
            self.assertFalse(report.incomplete)
        finally:
            registry.remove()

    def test_generic_unresolved_marker_serialises(self):
        City, Tag, Person = self.City, self.Tag, self.Person
        t1 = Tag(label="a").save()
        Person(name="Bob", favourite=t1).save()
        root = Person.objects.first()
        report = prefetch_related(root, max_depth=1, max_objects=0)
        self.assertTrue(report.incomplete)
        marker = root._data["favourite"]
        self.assertIsInstance(marker, UnresolvedReference)
        self.assertEqual(marker.document_type, Tag)
        payload = json.loads(root.to_json(json_options=json_util.DEFAULT_JSON_OPTIONS))
        self.assertEqual(payload["favourite"]["_cls"], "Tag")
        self.assertIn("$id", payload["favourite"]["_ref"])

    def test_missing_reference_reported(self):
        City, Tag, Person = self.City, self.Tag, self.Person
        from bson import DBRef, ObjectId

        person = Person(
            name="Bob",
            hometown=DBRef(City._get_collection_name(), ObjectId()),
        )
        person.save()
        root = Person.objects.first()
        report = prefetch_related(root, max_depth=1)
        self.assertTrue(any(v == "missing" for v in report.unresolved.values()))


@require_mongomock
class LazyFieldSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchlazytest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        class User(Document):
            name = StringField()
            meta = {"allow_inheritance": False}

        class Member(Document):
            name = StringField()
            pal = LazyReferenceField(User)
            meta = {"allow_inheritance": False}

        self.User, self.Member = User, Member
        User.drop_collection()
        Member.drop_collection()

    def test_lazy_reference_field_not_prefetched(self):
        User, Member = self.User, self.Member
        u = User(name="u").save()
        Member(name="m", pal=u).save()

        registry = CountingRegistry(User, Member).install()
        try:
            root = Member.objects.first()
            registry.reset()
            report = prefetch_related(root, max_depth=5)
            # LazyReferenceField must never be eagerly dereferenced.
            self.assertEqual(report.query_count, 0)
            self.assertIsInstance(root._data["pal"], LazyReference)
            # and it must stay untouched after prefetch
            self.assertNotIsInstance(root._data["pal"], UnresolvedReference)
        finally:
            registry.remove()


@require_mongomock
class SelectRelatedIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchintegrationtest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        class User(Document):
            name = StringField()
            meta = {"allow_inheritance": False}

        class Post(Document):
            title = StringField()
            author = ReferenceField(User)
            meta = {"allow_inheritance": False}

        self.User, self.Post = User, Post
        User.drop_collection()
        Post.drop_collection()

    def test_document_select_related_with_budget(self):
        User, Post = self.User, self.Post
        u = User(name="u").save()
        Post(title="p", author=u).save()
        post = Post.objects.first().select_related(PrefetchBudget(max_depth=1))
        report = post.get_prefetch_report()
        self.assertIsInstance(report, PrefetchReport)
        self.assertIs(last_report(), report)
        self.assertEqual(post.author.name, "u")

    def test_queryset_select_related_with_keyword_budget(self):
        User, Post = self.User, self.Post
        u1 = User(name="u1").save()
        u2 = User(name="u2").save()
        Post(title="p1", author=u1).save()
        Post(title="p2", author=u2).save()
        posts = Post.objects.select_related(max_depth=1, max_objects=1)
        self.assertIsInstance(posts, list)
        report = last_report()
        self.assertTrue(report.incomplete)
        self.assertEqual(report.objects_fetched, 1)
        markers = [
            p for p in posts if isinstance(p._data["author"], UnresolvedReference)
        ]
        self.assertEqual(len(markers), 1)
        self.assertIs(posts[0].get_prefetch_report(), report)

    def test_classic_select_related_unchanged(self):
        User, Post = self.User, self.Post
        u = User(name="u").save()
        Post(title="p", author=u).save()
        post = Post.objects.first().select_related()
        self.assertEqual(post.author.name, "u")
        posts = list(Post.objects.select_related(1))
        self.assertEqual([p.author.name for p in posts], ["u"])


@require_mongomock
class QueryCapMarkerTest(unittest.TestCase):
    """The per-round query limit delays (never drops) queries; combined with
    the object limit it must still leave identifiable markers behind."""

    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchqcaptest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def setUp(self):
        class A(Document):
            name = StringField()
            b = ReferenceField("B")
            meta = {"allow_inheritance": False}

        class B(Document):
            name = StringField()
            a = ReferenceField(A)
            meta = {"allow_inheritance": False}

        self.A, self.B = A, B
        A.drop_collection()
        B.drop_collection()
        a1 = A(name="a1"); a1.save()
        b1 = B(name="b1"); b1.save()
        a2 = A(name="a2"); a2.save()
        b2 = B(name="b2"); b2.save()
        a1.b = b1; a1.save()
        b1.a = a2; b1.save()
        a2.b = b2; a2.save()

    def test_rounds_never_exceed_cap_and_all_resolve(self):
        A, B = self.A, self.B
        registry = CountingRegistry(A, B).install()
        try:
            root = A.objects(name="a1").first()
            registry.reset()
            report = prefetch_related(root, max_depth=4, max_queries_per_round=1)
            self.assertEqual(report.queries_by_round, [1, 1, 1])
            self.assertEqual(registry.find_calls, 3)
            self.assertEqual(report.query_count, 3)
            self.assertEqual(report.objects_fetched, 3)
            self.assertFalse(report.incomplete)
            self.assertEqual(root.b.name, "b1")
            self.assertEqual(root.b.a.b.name, "b2")
        finally:
            registry.remove()

    def test_round_cap_records_queries_observed_by_stub(self):
        A, B = self.A, self.B
        registry = CountingRegistry(A, B).install()
        try:
            root = A.objects(name="a1").first()
            registry.reset()
            report = prefetch_related(root, max_depth=1, max_queries_per_round=1)
            self.assertEqual(report.query_count, registry.find_calls)
            # Two collections are reachable at depth 1 (b1 -> a2 discovered
            # only at depth 2, so depth 1 only needs B); assert one query max.
            self.assertLessEqual(max(report.queries_by_round), 1)
        finally:
            registry.remove()

    def test_zero_means_budget_disallows_everything(self):
        # A zero query cap combined with a finite depth would loop forever, so
        # callers are expected to use max_objects instead; zero object budget
        # is the supported "resolve nothing" switch.
        A = self.A
        root = A.objects(name="a1").first()
        report = prefetch_related(root, max_depth=1, max_objects=0)
        self.assertEqual(report.query_count, 0)
        self.assertTrue(report.incomplete)
        self.assertIsInstance(root._data["b"], UnresolvedReference)


@require_mongomock
class UnresolvedMarkerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disconnect_all()
        connect(
            db="prefetchmarkertest",
            host="mongodb://localhost",
            mongo_client_class=mongomock.MongoClient,
        )

    @classmethod
    def tearDownClass(cls):
        disconnect_all()

    def test_marker_is_copyable_and_picklable(self):
        import copy
        import pickle
        from bson import ObjectId

        class Loc(Document):
            name = StringField()
            meta = {"allow_inheritance": False}

        marker = UnresolvedReference(
            None, ObjectId(), Loc._get_collection_name(), "root[0].loc", "objects"
        )
        clone = copy.deepcopy(marker)
        self.assertIsInstance(clone, UnresolvedReference)
        self.assertEqual(clone.path, marker.path)
        self.assertEqual(clone.reason, "objects")
        self.assertEqual(clone, marker)
        self.assertIsNone(clone.document_type)

        restored = pickle.loads(pickle.dumps(marker))
        self.assertIsInstance(restored, UnresolvedReference)
        self.assertEqual(restored.path, marker.path)
        self.assertEqual(restored.id, marker.id)

    def test_marker_is_a_recognisable_lazy_reference(self):
        from bson import ObjectId

        marker = UnresolvedReference(
            None, ObjectId(), "widgets", "root[0].thing", "depth"
        )
        self.assertIsInstance(marker, LazyReference)
        self.assertEqual(marker.collection, "widgets")
        self.assertEqual(marker.path, "root[0].thing")
        self.assertIn("UnresolvedReference", repr(marker))
        self.assertIn("depth", str(marker))
