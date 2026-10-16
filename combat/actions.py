"""Actions an agent can choose in combat."""

from enum import Enum, auto


class Action(Enum):
    ATTACK = auto()
    SECOND_WIND = auto()
    ACTION_SURGE = auto()
    END_TURN = auto()
    CLEAVE = auto()
    TRIP = auto()
    DODGE = auto()
    HEAL = auto()
    ADVANCE = auto()
    DISENGAGE = auto()
