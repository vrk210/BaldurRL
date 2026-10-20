"""Registry of encounter seed ranges: which `env.reset(seed=...)` values were used, and for what.

Why this exists. Every evaluation, data collection and configuration choice draws
encounters from explicit seeds. Reusing a range that has already informed a
decision quietly turns a test set into a selection set. This happened once: the
M5/M6 held-out range 3000–7999 also guided the v1 → v2 retune. The registry is
code so that tools can enforce the one rule that must never slip, and anyone
can check a range before using it:

    python -m evaluation.seeds                    # ledger plus free gaps
    python -m evaluation.seeds --check 240000 5000

Rules:

- Reserve a new experiment's ranges here before running it. Experiment code
  itself may never merge.
- A range counts as used once any result from it has been looked at, even if
  the run was abandoned.
- Keep training, validation, selection and final-test ranges disjoint.
- FROZEN ranges are final-test sets. The evaluation CLIs refuse them unless
  `--final-test` is passed, and training or data collection never accepts them.
- Learner seeds (PPO `--seed`, agent RNG seeds) are separate from encounter
  seeds. `train.py` resets its training environments with `seed + rank`, which
  touches encounters 0–15 once each.

Overlapping USED entries are expected: the history records each use separately.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    USED = "used"
    RESERVED = "reserved"
    FROZEN = "frozen"


@dataclass(frozen=True)
class SeedRange:
    start: int
    stop: int  # exclusive, like range()
    status: Status
    use: str

    def overlaps(self, seeds: range) -> bool:
        return bool(seeds) and seeds.start < self.stop and self.start < seeds.stop

    def __str__(self) -> str:
        return f"{self.start}–{self.stop - 1} [{self.status.value}] {self.use}"


_U, _R, _F = Status.USED, Status.RESERVED, Status.FROZEN

REGISTRY: tuple[SeedRange, ...] = (
    SeedRange(0, 1000, _U, "Unit tests and smoke runs"),
    SeedRange(1000, 1200, _U, "PPO checkpoint selection during training (train.py default), all stages"),
    SeedRange(2000, 2200, _U, "M2 opening-action diagnostic"),
    SeedRange(3000, 4000, _U, "Original held-out evaluation, M0–M4"),
    SeedRange(3000, 8000, _U, "M5/M6 held-out evaluation; also informed the v1→v2 retune, so not an untouched test"),
    SeedRange(4000, 9000, _U, "Fresh 5,000-seed comparisons, M1A/M1B/M2"),
    SeedRange(4000, 5000, _U, "M4 policy diagnostic: threat-aware rule design"),
    SeedRange(5000, 10000, _U, "M4 policy diagnostic: paired evaluation"),
    SeedRange(9000, 14000, _U, "M2 exact optimum and fresh PPO comparison"),
    SeedRange(9000, 10000, _U, "M2 regret against the exact optimum"),
    SeedRange(10000, 15000, _U, "M4 expectimax final test"),
    SeedRange(18000, 18100, _U, "M4 rollout-planner pilot"),
    SeedRange(20000, 22000, _U, "M5/M6 stage design (development)"),
    SeedRange(20000, 40000, _U, "M4 value data, round 0"),
    SeedRange(20000, 42000, _U, "M2 value data"),
    SeedRange(40000, 42000, _U, "M4 value validation"),
    SeedRange(42000, 44000, _U, "M4 planner selection"),
    SeedRange(40000, 45000, _U, "M5/M6 review fresh-seed check"),
    SeedRange(51000, 51400, _U, "Transition-model tests and validation"),
    SeedRange(60000, 68000, _U, "M4 self-play value data"),
    SeedRange(100000, 120000, _U, "M5/M6 value-network data"),
    SeedRange(120000, 122000, _U, "M5/M6 value validation"),
    SeedRange(122000, 124000, _U, "M5/M6 planner configuration selection"),
    SeedRange(130000, 133000, _F, "M5/M6 final test; never used as of 2026-10-15"),
    SeedRange(140000, 200000, _U, "M5/M6 value data extension; guided-planning data and review checks"),
    SeedRange(210000, 210010, _U, "M5/M6 table transition benchmark"),
    SeedRange(210010, 212400, _U, "M5/M6 follow-up review: adversarial audit (211000–211029), env fingerprints"),
    SeedRange(220000, 220008, _U, "M5/M6 headroom probe: deterministic and smoke tests"),
    SeedRange(220008, 220010, _R, "Unused headroom development allocation"),
    SeedRange(220010, 220014, _U, "M5/M6 headroom throughput pilots"),
    SeedRange(220014, 221000, _R, "Unused headroom development allocation"),
    SeedRange(221000, 221012, _U, "M5/M6 headroom configuration selection A/B"),
    SeedRange(221012, 221100, _R, "Unused headroom selection allocation"),
    SeedRange(221100, 221112, _U, "M5/M6 headroom configuration selection C"),
    SeedRange(221112, 223000, _R, "Unused headroom selection allocation"),
    SeedRange(223000, 224000, _U, "Headroom: once-only paired confirmation, 1,000 encounters/stage; archive/simulator-research-2026-10-15:evaluation/headroom_probe_results.json"),
    SeedRange(224000, 230000, _R, "Unused headroom confirmation allocation and spare; no encounter resets"),
    SeedRange(230000, 230008, _U, "Simulator tactics mining: exploratory smoke tests (2026-10-15)"),
    SeedRange(230008, 232000, _R, "Unused simulator tactics exploration allocation"),
    SeedRange(232000, 236000, _U, "Simulator tactics: once-only paired confirmation, 13 fixed contrasts; archive/simulator-research-2026-10-15:evaluation/sim_mining.py"),
    SeedRange(236000, 240000, _R, "Unused tactics encounter spare; 236000–236127 used only as replay combat RNG seeds, not encounter resets"),
)


def overlapping(seeds: range) -> tuple[SeedRange, ...]:
    """Registry entries that share at least one seed with ``seeds``."""
    return tuple(entry for entry in REGISTRY if entry.overlaps(seeds))


def require_not_frozen(seeds: Sequence[int] | range, *, final_test: bool = False) -> None:
    """Refuse a frozen final-test range unless the caller explicitly runs the final test."""
    span = seeds if isinstance(seeds, range) else range(min(seeds), max(seeds) + 1) if len(seeds) else range(0)
    frozen = [entry for entry in overlapping(span) if entry.status is Status.FROZEN]
    if frozen and not final_test:
        raise ValueError(
            f"Seeds {span.start}–{span.stop - 1} overlap the frozen final-test range {frozen[0]}. "
            "Use it once, for the final table only, by passing --final-test."
        )


def free_ranges(low: int = 0, high: int = 300_000) -> tuple[range, ...]:
    """Gaps in [low, high) not covered by any registry entry."""
    covered = sorted((max(e.start, low), min(e.stop, high)) for e in REGISTRY if e.stop > low and e.start < high)
    gaps, cursor = [], low
    for start, stop in covered:
        if start > cursor:
            gaps.append(range(cursor, start))
        cursor = max(cursor, stop)
    if cursor < high:
        gaps.append(range(cursor, high))
    return tuple(gaps)


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser(description="Show the encounter seed registry or check a range against it")
    parser.add_argument("--check", nargs=2, type=int, metavar=("START", "COUNT"))
    args = parser.parse_args(argv)
    if args.check is not None:
        start, count = args.check
        if count < 1:
            parser.error("COUNT must be positive")
        hits = overlapping(range(start, start + count))
        for entry in hits:
            print(entry)
        if not hits:
            print(f"{start}–{start + count - 1} is unallocated; reserve it in evaluation/seeds.py before use")
        return 1 if any(entry.status is not Status.USED for entry in hits) else 0
    for entry in REGISTRY:
        print(entry)
    print("Free gaps:", ", ".join(f"{gap.start}–{gap.stop - 1}" for gap in free_ranges(100_000) if len(gap) >= 1000))
    return 0


if __name__ == "__main__":
    sys.exit(main())
