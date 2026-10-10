"""Leaf value functions: batches of observations to estimated win probabilities.

`RaceValue` is a cheap hand-built heuristic (a deterministic mean-field damage
race squashed by a two-parameter logistic). `MLPValue` runs a small trained
multilayer perceptron in NumPy. Both read observations through the stage's
published field names and never step the simulator. Expected damage per attack
comes from the exact distribution helper in `combat.mechanics`, so no dice
formula is restated here.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from combat.damage import DamageSpec
from combat.distributions import attack_damage_distribution, second_wind_healing_distribution
from combat.stages import make_env


_ALLY_COUNTS = ("hp", "action_count", "bonus_action_count", "second_wind_count", "action_surge_count", "cleave_count")
_ENEMY_STATS = ("hp", "max_hp", "ac", "attack_bonus", "damage_die_size", "damage_bonus")


class _Layout:
    """Column indices of ally and enemy fields for one stage, plus preset stats."""

    def __init__(self, stage: str) -> None:
        env = make_env(stage)
        env.reset(seed=0)
        fields = list(env.observation_fields)
        self.stage = stage
        self.high = np.asarray(env.observation_space.high, dtype=np.float64)
        roster = env._roster()
        self.ally = roster.allies[0]
        self.n_allies = len(roster.allies)
        self.n_enemies = len(roster.enemies)
        self.preset_enemy = roster.enemies[0] if stage == "m0" else None
        env.close()

        def column(name: str) -> int:
            return fields.index(name) if name in fields else -1

        def ally_prefix(slot: int) -> str:
            return f"ally_{slot}_" if f"ally_{slot}_hp" in fields else "fighter_"

        def enemy_prefix(slot: int) -> str:
            if f"enemy_{slot}_hp" in fields:
                return f"enemy_{slot}_"
            return "enemy_" if "enemy_hp" in fields else "goblin_"

        self.ally_cols = np.array([[column(ally_prefix(s) + f) for f in _ALLY_COUNTS] for s in range(self.n_allies)])
        self.enemy_cols = np.array([[column(enemy_prefix(s) + f) for f in _ENEMY_STATS] for s in range(self.n_enemies)])
        self.actor_col = column("active_actor_index")
        self.round_col = column("round_number")

    def allies(self, obs: np.ndarray) -> np.ndarray:
        """(n, allies, 6) array; absent fields (e.g. Cleave before M2) read as 0."""
        out = np.zeros((len(obs), self.n_allies, len(_ALLY_COUNTS)))
        valid = self.ally_cols >= 0
        out[:, valid] = obs[:, self.ally_cols[valid]]
        return out

    def enemies(self, obs: np.ndarray) -> np.ndarray:
        """(n, enemies, 6) array of hp, max_hp, ac, attack, die, bonus."""
        out = np.zeros((len(obs), self.n_enemies, len(_ENEMY_STATS)))
        valid = self.enemy_cols >= 0
        out[:, valid] = obs[:, self.enemy_cols[valid]]
        if self.preset_enemy is not None:
            e = self.preset_enemy
            out[:, 0, 1:] = (e.max_hp, e.armor_class, e.attack_bonus, e.damage.die_size, e.damage.bonus)
        return out

    def actor(self, obs: np.ndarray) -> np.ndarray:
        return obs[:, self.actor_col].astype(int) if self.actor_col >= 0 else np.zeros(len(obs), dtype=int)


@lru_cache(maxsize=None)
def _mean_damage(attack_bonus: int, armor_class: int, dice: int, die: int, bonus: int, divisor: int = 1) -> float:
    distribution = attack_damage_distribution(attack_bonus, armor_class, DamageSpec(dice, die, bonus), divisor)
    return float(sum(amount * p for amount, p in distribution.items()))


@lru_cache(maxsize=None)
def _expected_heal(deficit: int) -> float:
    return float(sum(min(deficit, heal) * p for heal, p in second_wind_healing_distribution().items()))


class RaceValue:
    """Heuristic leaf: deterministic mean-field race between the two sides.

    Allies (with expected Second Wind healing folded into HP) spend their
    attacks round by round on the living enemy with the lowest
    attacks-to-kill per unit threat; Action Surge adds one attack and Cleave
    deals expected halved damage to every living enemy. Enemies deal their
    expected damage per round to the lowest living ally slot. Only allies at or
    after the active slot still act in the current round. The race returns a
    margin in [-1, 1] (surviving ally HP share on a win, minus surviving enemy
    HP share on a loss), mapped to a probability by `sigmoid(scale * margin +
    bias)`. Fit `scale` and `bias` per stage with `value_training.py fit-race`
    (M4: 2.9757 and 0.0423; M2: 2.7834 and -0.0003); the defaults are
    uncalibrated placeholders.
    """

    def __init__(self, stage: str, *, scale: float = 4.0, bias: float = 0.0, max_rounds: int = 40) -> None:
        self.layout = _Layout(stage)
        self.scale = scale
        self.bias = bias
        self.max_rounds = max_rounds

    def margin(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float64).reshape(-1, len(self.layout.high))
        n = len(obs)
        layout = self.layout
        ally = layout.ally
        allies = layout.allies(obs)
        enemies = layout.enemies(obs)
        ally_hp = allies[:, :, 0].copy()
        alive = ally_hp > 0
        deficits = np.clip(ally.max_hp - ally_hp, 0, ally.max_hp).astype(int)
        heal = np.vectorize(_expected_heal)(deficits) * allies[:, :, 3] * alive
        ally_hp = ally_hp + heal
        enemy_hp = enemies[:, :, 0].copy()
        ints = enemies.astype(int)
        hit_damage = np.vectorize(lambda ac: _mean_damage(ally.attack_bonus, ac, ally.damage.dice_count, ally.damage.die_size, ally.damage.bonus))(ints[:, :, 2])
        cleave_damage = np.vectorize(lambda ac: _mean_damage(ally.attack_bonus, ac, ally.damage.dice_count, ally.damage.die_size, ally.damage.bonus, 2))(ints[:, :, 2])
        threat = np.vectorize(lambda atk, die, bonus: _mean_damage(atk, ally.armor_class, 1, die, bonus))(ints[:, :, 3], ints[:, :, 4], ints[:, :, 5])
        # Cleave: each charge trades one ordinary attack for halved damage to all living enemies.
        cleaves = (allies[:, :, 5] * alive).sum(axis=1)
        enemy_hp = np.maximum(enemy_hp - cleaves[:, None] * cleave_damage * (enemy_hp > 0), 0.0)
        bonus_attacks = (allies[:, :, 4] * alive).sum(axis=1) - cleaves
        start_hp_allies = np.maximum(ally_hp.sum(axis=1), 1e-9)
        start_hp_enemies = np.maximum(enemies[:, :, 0].sum(axis=1), 1e-9)
        priority = np.where(enemy_hp > 0, enemy_hp / np.maximum(hit_damage, 1e-9) / np.maximum(threat, 1e-9), np.inf)
        order = np.argsort(priority, axis=1)
        actor = layout.actor(obs)
        slots = np.arange(layout.n_allies)[None, :]
        first_round = (slots >= actor[:, None]) & alive
        result = np.zeros(n)
        done = np.zeros(n, dtype=bool)
        for round_index in range(self.max_rounds):
            acting = first_round if round_index == 0 else (ally_hp > 0)
            budget = acting.sum(axis=1).astype(np.float64) + np.where(round_index == 0, bonus_attacks, 0.0)
            budget = np.maximum(budget, 0.0)
            for rank in range(layout.n_enemies):
                target = order[:, rank]
                hp = enemy_hp[np.arange(n), target]
                damage = hit_damage[np.arange(n), target]
                used = np.minimum(budget, hp / np.maximum(damage, 1e-9))
                enemy_hp[np.arange(n), target] = np.maximum(hp - used * damage, 0.0)
                budget = budget - used
            won = ~done & (enemy_hp.sum(axis=1) <= 1e-9)
            result[won] = ally_hp[won].clip(min=0).sum(axis=1) / start_hp_allies[won]
            done |= won
            incoming = (threat * (enemy_hp > 0)).sum(axis=1)
            for slot in range(layout.n_allies):
                hp = ally_hp[:, slot]
                taken = np.minimum(np.maximum(hp, 0.0), incoming)
                ally_hp[:, slot] = hp - taken
                incoming = incoming - taken
            lost = ~done & (ally_hp.clip(min=0).sum(axis=1) <= 1e-9)
            result[lost] = -enemy_hp[lost].sum(axis=1) / start_hp_enemies[lost]
            done |= lost
            if done.all():
                break
        return result

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.scale * self.margin(observations) + self.bias)))


class MLPValue:
    """NumPy inference for a trained value MLP saved by `value_training.py`.

    Inputs are observations divided by the stage's observation-space highs.
    Inactive allies' Action and Bonus Action counts are zeroed first: both are
    set (not added) at the start of that ally's next turn, so they cannot affect
    the outcome, and zeroing keeps search leaves on the training distribution.
    """

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]], stage: str, *, canonicalize: bool = True) -> None:
        self.layout = _Layout(stage)
        self.weights = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in weights]
        self.canonicalize = canonicalize

    @classmethod
    def load(cls, path: str | Path) -> "MLPValue":
        path = Path(path)
        data = np.load(path)
        meta = json.loads(str(data["meta"]))
        layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(meta["layers"])]
        if meta.get("feature_set", "obs") != "obs":
            raise ValueError("Only observation-feature value networks are supported")
        return cls(layers, meta["stage"], canonicalize=meta.get("canonicalize", True))

    def features(self, observations: np.ndarray) -> np.ndarray:
        return value_features(self.layout, observations, self.canonicalize)

    def logits(self, observations: np.ndarray) -> np.ndarray:
        x = self.features(observations)
        for index, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if index < len(self.weights) - 1:
                x = np.maximum(x, 0.0)
        return x[:, 0]

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logits(observations)))


class EnsembleValue:
    """Average of several value functions' logits (reduces exploitable noise)."""

    def __init__(self, members: list[MLPValue]) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = members

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        logits = np.mean([member.logits(observations) for member in self.members], axis=0)
        return 1.0 / (1.0 + np.exp(-logits))


def value_features(layout: _Layout, observations: np.ndarray, canonicalize: bool = True) -> np.ndarray:
    """Observations divided by the observation-space highs, optionally canonicalized."""
    obs = np.asarray(observations, dtype=np.float64).reshape(-1, len(layout.high)).copy()
    if canonicalize and layout.actor_col >= 0:
        actor = layout.actor(obs)
        for slot in range(layout.n_allies):
            inactive = actor != slot
            for field in (1, 2):  # action_count, bonus_action_count
                column = layout.ally_cols[slot, field]
                if column >= 0:
                    obs[inactive, column] = 0.0
    return obs / layout.high


def make_layout(stage: str) -> _Layout:
    return _Layout(stage)
