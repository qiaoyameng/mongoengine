from dataclasses import dataclass, field


__all__ = (
    "PrefetchBudget",
    "PrefetchReport",
    "UnresolvedReference",
)


@dataclass(frozen=True)
class PrefetchBudget:
    max_depth: int | None = 1
    max_unique_objects: int | None = None
    max_queries_per_round: int | None = None

    def __post_init__(self):
        limits = (
            ("max_depth", self.max_depth),
            ("max_unique_objects", self.max_unique_objects),
            ("max_queries_per_round", self.max_queries_per_round),
        )
        for name, value in limits:
            if value is not None:
                if not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer or None")
                if value < 0:
                    raise ValueError(f"{name} must be greater than or equal to 0")


@dataclass(frozen=True)
class UnresolvedReference:
    path: str
    collection: str
    document_id: object
    document_class: str | None
    depth: int
    reason: str


@dataclass(frozen=True)
class PrefetchReport:
    max_depth: int | None
    max_unique_objects: int | None
    max_queries_per_round: int | None
    queries_executed: int
    unique_objects_fetched: int
    depths_reached: int
    truncated_by: str | None
    unresolved: tuple[UnresolvedReference, ...] = field(default_factory=tuple)

    @property
    def unresolved_paths(self):
        return tuple(item.path for item in self.unresolved)

    @property
    def complete(self):
        return not self.unresolved

    def as_dict(self):
        return {
            "max_depth": self.max_depth,
            "max_unique_objects": self.max_unique_objects,
            "max_queries_per_round": self.max_queries_per_round,
            "queries_executed": self.queries_executed,
            "unique_objects_fetched": self.unique_objects_fetched,
            "depths_reached": self.depths_reached,
            "truncated_by": self.truncated_by,
            "unresolved": [
                {
                    "path": item.path,
                    "collection": item.collection,
                    "document_id": item.document_id,
                    "document_class": item.document_class,
                    "depth": item.depth,
                    "reason": item.reason,
                }
                for item in self.unresolved
            ],
        }
