"""Character state for combat."""

from dataclasses import dataclass


@dataclass
class Character:
    name: str
    max_hp: int
    hp: int
    armor_class: int
    attack_bonus: int
    damage_dice_count: int
    damage_die_size: int
    damage_bonus: int

    @property
    def alive(self) -> bool:
        return self.hp > 0


@dataclass
class Fighter(Character):
    action_available: bool = True
    bonus_action_available: bool = True
    second_wind_available: bool = True
    action_surge_available: bool = True


@dataclass
class Goblin(Character):
    pass
