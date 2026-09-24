"""Counts for the resources used by M0 abilities."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Mapping


class Resource(Enum):
    ACTION = auto()
    BONUS_ACTION = auto()
    SECOND_WIND = auto()
    ACTION_SURGE = auto()


@dataclass
class ResourcePool:
    _amounts: dict[Resource, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        amounts = dict(self._amounts)
        if any(amount < 0 for amount in amounts.values()):
            raise ValueError("Resource counts cannot be negative")
        self._amounts = amounts

    def get(self, resource: Resource) -> int:
        return self._amounts.get(resource, 0)

    def has(self, costs: Mapping[Resource, int]) -> bool:
        if any(cost < 0 for cost in costs.values()):
            raise ValueError("Resource costs cannot be negative")
        return all(self.get(resource) >= cost for resource, cost in costs.items())

    def spend(self, costs: Mapping[Resource, int]) -> None:
        if not self.has(costs):
            raise ValueError("Insufficient resources")
        for resource, cost in costs.items():
            self._amounts[resource] = self.get(resource) - cost

    def gain(self, resource: Resource, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Resource gains cannot be negative")
        self._amounts[resource] = self.get(resource) + amount

    def set(self, resource: Resource, amount: int) -> None:
        if amount < 0:
            raise ValueError("Resource counts cannot be negative")
        self._amounts[resource] = amount
