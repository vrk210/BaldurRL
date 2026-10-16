"""Compare deterministic semantic episode fingerprints against a Git base.

Usage: python scripts/compare_refactor.py --base COMMIT
Uses existing smoke seeds 0..99, not fresh research/final-test encounters.
The same worker runs both revisions with the current Python/dependencies.
Observations, masks, actions, rewards, flags and diagnostic game facts are
hashed in order; timing and paths are never included. No trained models run.
This checks regression equivalence, not agreement with real BG3 mechanics.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import zipfile


def fingerprints(root: Path) -> dict[str, dict[str, str | int]]:
    sys.path.insert(0, str(root))
    import numpy as np
    from agents.heuristic_agent import make_heuristic_agent
    from agents.random_agent import RandomAgent
    from combat.stages import make_env

    def numpy_value(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Unsupported trace value: {type(value)}")

    results = {}
    for stage in ("m0", "m1a", "m1b", "m2", "m3", "m4", "m5", "m6"):
        variants = ["random", "heuristic"]
        if stage in ("m5", "m6"):
            variants.append("tuned" if stage == "m5" else "adaptive")
        for variant in variants:
            digest = hashlib.sha256()
            decisions = 0
            for seed in range(100):
                env = make_env(stage)
                policy = (RandomAgent(seed=seed) if variant == "random" else
                          make_heuristic_agent(stage, None if variant == "heuristic" else variant))
                observation, info = env.reset(seed=seed)
                digest.update(json.dumps([seed, observation, info], default=numpy_value, sort_keys=True).encode())
                while True:
                    mask = env.action_masks()
                    action = policy.choose_action(observation.copy(), mask.copy())
                    observation, reward, terminated, truncated, info = env.step(action)
                    digest.update(json.dumps([mask, action, observation, reward, terminated, truncated, info],
                                             default=numpy_value, sort_keys=True).encode())
                    decisions += 1
                    if terminated or truncated:
                        break
                env.close()
            results[f"{stage}/{variant}"] = {"episodes": 100, "decisions": decisions, "sha256": digest.hexdigest()}
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git commit whose runtime behavior must be preserved")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(fingerprints(args.worker.resolve()), sort_keys=True))
        return 0
    if not args.base:
        parser.error("--base is required")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}"], cwd=root, text=True).strip()
    archive = subprocess.check_output(["git", "archive", "--format=zip", revision], cwd=root)
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(prefix="baldurrl-refactor-") as directory:
        baseline = Path(directory)
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            bundle.extractall(baseline)
        before = json.loads(subprocess.check_output([sys.executable, script, "--worker", str(baseline)], cwd=baseline, text=True))
        after = json.loads(subprocess.check_output([sys.executable, script, "--worker", str(root)], cwd=root, text=True))
    mismatches = [name for name in before.keys() | after.keys() if before.get(name) != after.get(name)]
    print(json.dumps({"base": revision, "fingerprints": after, "mismatches": sorted(mismatches)}, indent=2))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
