"""The seed registry is well formed and the CLIs refuse frozen final-test seeds."""

import sys

import pytest

from evaluation import seeds
from evaluation.seeds import REGISTRY, Status, free_ranges, overlapping, require_not_frozen

FROZEN = next(entry for entry in REGISTRY if entry.status is Status.FROZEN)


def test_entries_are_ordered_ranges_and_reservations_are_exclusive():
    for entry in REGISTRY:
        assert 0 <= entry.start < entry.stop and entry.use
    exclusive = [entry for entry in REGISTRY if entry.status is not Status.USED]
    for entry in exclusive:
        others = [o for o in overlapping(range(entry.start, entry.stop)) if o is not entry]
        assert not others, f"{entry} overlaps {others}"


def test_frozen_range_requires_explicit_final_test():
    inside = range(FROZEN.start + 10, FROZEN.start + 20)
    with pytest.raises(ValueError, match="frozen"):
        require_not_frozen(inside)
    with pytest.raises(ValueError, match="frozen"):
        require_not_frozen([FROZEN.stop - 1])
    require_not_frozen(inside, final_test=True)
    require_not_frozen(range(FROZEN.stop, FROZEN.stop + 100))
    require_not_frozen(range(0))


def test_free_ranges_avoid_every_entry():
    for gap in free_ranges(0, 300_000):
        assert not overlapping(gap)


def test_check_cli_flags_frozen_and_unallocated(capsys):
    assert seeds.main(["--check", str(FROZEN.start), "10"]) == 1
    gap = free_ranges(100_000)[0]
    assert seeds.main(["--check", str(gap.start), "10"]) == 0
    assert "unallocated" in capsys.readouterr().out


def test_evaluate_and_train_clis_refuse_frozen_seeds(monkeypatch):
    import evaluation.evaluate as evaluate
    import train

    monkeypatch.setattr(sys, "argv", ["evaluate", "--stage", "m5", "--agent", "heuristic",
                                      "--seed-start", str(FROZEN.start), "--episodes", "5"])
    with pytest.raises(SystemExit):
        evaluate.main()
    with pytest.raises(SystemExit):
        train.parse_args(["--stage", "m5", "--eval-seed-start", str(FROZEN.start)])
