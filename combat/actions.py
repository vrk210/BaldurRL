"""Actions an agent can choose in combat."""

from enum import Enum, auto


class Action(Enum):
    ATTACK = auto()
    SECOND_WIND = auto()
    ACTION_SURGE = auto()
    END_TURN = auto()
