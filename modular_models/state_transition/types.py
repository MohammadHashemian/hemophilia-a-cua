from __future__ import annotations

from enum import IntEnum, StrEnum


class Strategy(StrEnum):
    PROPHYLAXIS = "prophylaxis"
    ON_DEMAND = "on_demand"


class ChronicState(IntEnum):
    NO_MINIMAL_ARTHROPATHY = 0
    MILD_ARTHROPATHY = 1
    MODERATE_ARTHROPATHY = 2
    SEVERE_ARTHROPATHY = 3
    DEATH = 4


class DeathCause(IntEnum):
    ALIVE = 0
    BACKGROUND = 1
    ICH = 2


class AcuteEvent(StrEnum):
    JOINT_BLEED = "joint_bleed"
    NON_MAJOR_NON_JOINT = "non_major_non_joint_bleed"
    NON_ICH_MAJOR = "non_ich_major_bleed"
    ICH = "intracranial_hemorrhage"


EVENT_ORDER: tuple[AcuteEvent, ...] = (
    AcuteEvent.JOINT_BLEED,
    AcuteEvent.NON_MAJOR_NON_JOINT,
    AcuteEvent.NON_ICH_MAJOR,
    AcuteEvent.ICH,
)
