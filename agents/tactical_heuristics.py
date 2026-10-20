"""Hand-written M5/M6 policies: the naive M4-style baseline and smarter variants.

Policies read only the observation and action mask and return an index. They
encode play heuristics (priorities and rough expected-damage estimates), not
combat rules; the environment owns every mechanic. Fighter offense (+5,
1d8+3) and AC 16 are the documented preset constants, which the observation
omits, as the M0–M4 heuristics also assume.
"""

from dataclasses import dataclass

import numpy as np

from combat.actions import Action
from combat.tactical_env import (
    N_ALLIES,
    N_ENEMIES,
    STAGE_RULES,
    tactical_decisions,
    tactical_observation_fields,
)


_FIGHTER_AC = 16
_FIGHTER_ATTACK = 5
_PRIORITIES = ("lowest_hp", "role")
# The adaptive variant dives once the sturdiest front-rank enemy's HP divided by
# our hit chance falls below this.
_ADAPTIVE_DIVE_DURABILITY = 42.0
_ADVANCE_MODES = ("never", "all", "adaptive")


@dataclass(frozen=True)
class TacticalHeuristicConfig:
    """Knobs for the M5/M6 heuristic family; defaults reproduce the naive M4 rules."""

    trip: bool = False
    priority: str = "lowest_hp"
    cleave_min_targets: int = 2
    count_reachable_for_cleave: bool = False
    advance: str = "never"
    second_wind_threshold: int = 10

    def __post_init__(self) -> None:
        if self.priority not in _PRIORITIES:
            raise ValueError(f"Unknown priority {self.priority!r}")
        if self.advance not in _ADVANCE_MODES:
            raise ValueError(f"Unknown advance mode {self.advance!r}")


@dataclass(frozen=True)
class _Ally:
    slot: int
    hp: int
    action: int
    bonus: int
    deep: bool

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass(frozen=True)
class _Enemy:
    slot: int
    hp: int
    ac: int
    attack: int
    die: int
    bonus: int
    role: str
    heal_count: int
    back_rank: bool

    @property
    def alive(self) -> bool:
        return self.hp > 0


def expected_damage(attack_bonus: int, die: int, bonus: int, armor_class: int) -> float:
    """Policy-side estimate of one attack's mean damage (not used by the simulator)."""
    p_hit = min(0.95, max(0.05, (21 + attack_bonus - armor_class) / 20))
    p_crit = 0.05
    mean_die = (die + 1) / 2
    return p_hit * (mean_die + bonus) + p_crit * mean_die


class TacticalHeuristicAgent:
    """Priority policy for M5/M6 controlled by a ``TacticalHeuristicConfig``."""

    def __init__(self, stage: str, config: TacticalHeuristicConfig | None = None) -> None:
        self.config = config or TacticalHeuristicConfig()
        self.ranks = stage == "m6"
        heal = STAGE_RULES[stage].heal
        self._heal_value = heal.dice_count * (heal.die_size + 1) / 2 + heal.bonus  # Mean heal.
        fields = tactical_observation_fields(stage)
        self._index = {name: position for position, name in enumerate(fields)}
        self._decision_index = {
            (decision.action, decision.target_index): position
            for position, decision in enumerate(tactical_decisions(stage))
        }

    # --- observation parsing -----------------------------------------------------

    def _value(self, observation: np.ndarray, name: str) -> int:
        return int(observation[self._index[name]])

    def _ally(self, observation: np.ndarray, slot: int) -> _Ally:
        get = lambda field: self._value(observation, f"ally_{slot}_{field}")  # noqa: E731
        return _Ally(
            slot=slot, hp=get("hp"), action=get("action_count"), bonus=get("bonus_action_count"),
            deep=bool(get("deep")) if self.ranks else False,
        )

    def _enemy(self, observation: np.ndarray, slot: int) -> _Enemy:
        get = lambda field: self._value(observation, f"enemy_{slot}_{field}")  # noqa: E731
        role = next(name for name in ("brute", "archer", "healer") if get(f"is_{name}"))
        return _Enemy(
            slot=slot, hp=get("hp"), ac=get("ac"), attack=get("attack_bonus"),
            die=get("damage_die_size"), bonus=get("damage_bonus"), role=role,
            heal_count=get("heal_count"),
            back_rank=bool(get("back_rank")) if self.ranks else False,
        )

    def _decision(self, action: Action, target: int | None = None) -> int:
        return self._decision_index[(action, target)]

    def _legal(self, mask: np.ndarray, action: Action, target: int | None = None) -> bool:
        return bool(mask[self._decision(action, target)])

    # --- policy estimates ----------------------------------------------------------

    def _threat(self, enemy: _Enemy) -> float:
        attack = expected_damage(enemy.attack, enemy.die, enemy.bonus, _FIGHTER_AC)
        if enemy.role == "healer" and enemy.heal_count > 0:
            return max(attack, self._heal_value)
        return attack

    def _priority_key(self, enemy: _Enemy) -> tuple[float, ...]:
        """Smaller keys are attacked first."""
        if self.config.priority == "role":
            if enemy.role == "healer":
                rank = 0 if enemy.heal_count > 0 else 2
            else:
                rank = 1 if enemy.role == "archer" else 3
            return (rank, enemy.hp, enemy.slot)
        return (enemy.hp, enemy.slot)

    # --- decisions ---------------------------------------------------------------

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        mask = np.asarray(action_mask, dtype=np.bool_)
        if not mask.any():
            raise ValueError("No legal actions available")
        config = self.config
        allies = [self._ally(observation, slot) for slot in range(N_ALLIES)]
        enemies = [self._enemy(observation, slot) for slot in range(N_ENEMIES)]
        me = allies[self._value(observation, "active_actor_index")]

        if self._legal(mask, Action.SECOND_WIND) and me.hp <= config.second_wind_threshold:
            return self._decision(Action.SECOND_WIND)
        move = self._movement(mask, allies, enemies)
        if move is not None:
            return move

        targets = [enemy for enemy in enemies if self._legal(mask, Action.ATTACK, enemy.slot)]
        focus = min(targets, key=self._priority_key) if targets else None
        trip = self._trip_choice(mask, focus)
        if trip is not None:
            return trip
        if self._legal(mask, Action.CLEAVE) and self._cleave_count(targets, enemies) >= config.cleave_min_targets:
            return self._decision(Action.CLEAVE)
        if focus is not None:
            return self._decision(Action.ATTACK, focus.slot)
        if self._legal(mask, Action.ACTION_SURGE) and me.action == 0:
            return self._decision(Action.ACTION_SURGE)
        if config.trip:
            trippable = [enemy for enemy in enemies if self._legal(mask, Action.TRIP, enemy.slot)]
            if trippable:
                return self._decision(Action.TRIP, max(trippable, key=lambda e: (self._threat(e), -e.slot)).slot)
        if self._legal(mask, Action.END_TURN):
            return self._decision(Action.END_TURN)
        return int(np.flatnonzero(mask)[0])

    def _trip_choice(self, mask: np.ndarray, focus: _Enemy | None) -> int | None:
        """Trip the focus target before attacking it."""
        if not self.config.trip or focus is None:
            return None
        if self._legal(mask, Action.TRIP, focus.slot):
            return self._decision(Action.TRIP, focus.slot)
        return None

    def _cleave_count(self, targets: list[_Enemy], enemies: list[_Enemy]) -> int:
        if self.config.count_reachable_for_cleave:
            return len(targets)
        return sum(enemy.alive for enemy in enemies)

    def _movement(self, mask: np.ndarray, allies: list[_Ally], enemies: list[_Enemy]) -> int | None:
        config = self.config
        if config.advance == "never" or not self._legal(mask, Action.ADVANCE):
            return None
        if (
            config.advance == "adaptive"
            and not any(ally.alive and ally.deep for ally in allies)  # commit once anyone dove
            and self._front_durability(enemies) < _ADAPTIVE_DIVE_DURABILITY
        ):
            return None
        return self._decision(Action.ADVANCE)

    def _front_durability(self, enemies: list[_Enemy]) -> float:
        """HP of the sturdiest living front-rank enemy divided by our hit chance."""
        def hits_needed(enemy: _Enemy) -> float:
            p_hit = min(0.95, max(0.05, (21 + _FIGHTER_ATTACK - enemy.ac) / 20))
            return enemy.hp / p_hit
        return max((hits_needed(e) for e in enemies if e.alive and not e.back_rank), default=0.0)


# Named variants evaluated in docs/RESULTS.md (M5/M6 section). "naive" applies
# the M4 priorities and ignores every M5/M6 option; the others were developed on
# development seeds 20000+ (never the held-out range).
_TUNED = {"cleave_min_targets": 3, "second_wind_threshold": 12}
VARIANTS: dict[str, dict[str, TacticalHeuristicConfig]] = {
    "m5": {
        "naive": TacticalHeuristicConfig(),
        "trip_focus": TacticalHeuristicConfig(trip=True),
        "healer_first": TacticalHeuristicConfig(trip=True, priority="role"),
        "tuned": TacticalHeuristicConfig(trip=True, **_TUNED),
    },
    "m6": {
        "naive": TacticalHeuristicConfig(),
        "hold": TacticalHeuristicConfig(trip=True, count_reachable_for_cleave=True, **_TUNED),
        "dive_all": TacticalHeuristicConfig(
            trip=True, count_reachable_for_cleave=True, advance="all", **_TUNED
        ),
        "adaptive": TacticalHeuristicConfig(
            trip=True, count_reachable_for_cleave=True, advance="adaptive", **_TUNED
        ),
    },
}


def make_tactical_heuristic(stage: str, variant: str = "naive") -> TacticalHeuristicAgent:
    try:
        config = VARIANTS[stage][variant]
    except KeyError as exc:
        raise ValueError(f"Unknown {stage} heuristic variant: {variant}") from exc
    return TacticalHeuristicAgent(stage, config)
