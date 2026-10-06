"""Deterministic turn ordering without dice, HP, rewards, or policies."""

from collections.abc import Callable, Sequence

from .actors import ActorRef


class TurnManager:
    """Ordered actor refs with round tracking and dead-actor skipping.

    Answers whose turn it is, who comes next, and whether advancing wraps
    to a new round. Never selects actions, rolls dice, mutates HP,
    calculates rewards, or executes attacks.
    """

    def __init__(
        self, order: Sequence[ActorRef], round_number: int = 1, *, current: ActorRef | None = None
    ) -> None:
        items = tuple(order)
        if not items:
            raise ValueError("Turn order must be non-empty")
        if len(set(items)) != len(items):
            raise ValueError("Turn order must not contain duplicates")
        if round_number < 1:
            raise ValueError("Round number starts at 1")
        self._order = items
        if current is not None and current not in items:
            raise ValueError("Current actor must belong to the turn order")
        self._current_index = 0 if current is None else items.index(current)
        self._round_number = round_number

    @property
    def order(self) -> tuple[ActorRef, ...]:
        return self._order

    @property
    def current(self) -> ActorRef:
        return self._order[self._current_index]

    @property
    def round_number(self) -> int:
        return self._round_number

    def peek_next(self, is_alive: Callable[[ActorRef], bool]) -> tuple[ActorRef | None, bool]:
        """Return (next living ref, would_wrap) without mutating state."""
        found = self._find_next_living(is_alive)
        if found is None:
            return None, False
        next_index, would_wrap = found
        return self._order[next_index], would_wrap

    def advance(self, is_alive: Callable[[ActorRef], bool]) -> bool:
        """Move to the next living actor; return True when wrapping to a new round.

        Skips dead actors. When no actor is alive, stays put, keeps the
        round number, and returns False instead of looping forever.
        """
        found = self._find_next_living(is_alive)
        if found is None:
            return False
        next_index, wrapped = found
        self._current_index = next_index
        if wrapped:
            self._round_number += 1
        return wrapped

    def _find_next_living(
        self, is_alive: Callable[[ActorRef], bool]
    ) -> tuple[int, bool] | None:
        count = len(self._order)
        for step in range(1, count + 1):
            candidate = (self._current_index + step) % count
            if is_alive(self._order[candidate]):
                return candidate, candidate <= self._current_index
        return None
