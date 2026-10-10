"""Collect value-function data and fit a small win-probability MLP.

collect: play seeded episodes with a policy and save every decision
observation with the episode outcome (and, for the expectimax planner, the
search root value at that decision).

    python value_training.py collect --stage m4 --policy threat --seed-start 20000 --episodes 4000 --out runs/value/data/threat_20000.npz
    python value_training.py collect --stage m4 --policy expectimax --leaf mlp:runs/value/v0.npz --depth 1 --seed-start 60000 --episodes 2000 --out runs/value/data/x1_60000.npz

fit: regress V(observation) -> P(win) by binary cross-entropy on soft targets.

    python value_training.py fit --stage m4 --train runs/value/data/*.npz --val runs/value/val/*.npz --out runs/value/v0.npz
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np

from agents.policy import Policy
from agents.value_functions import RaceValue, make_layout, value_features
from combat.stages import make_env
from evaluation.planning_eval import make_policy


class EpsilonPolicy:
    """With probability epsilon, a uniformly random legal action (coverage only)."""

    def __init__(self, base: Policy, epsilon: float, seed: int) -> None:
        self.base = base
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def choose_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(action_mask)))
        return self.base.choose_action(observation, action_mask)


def collect(stage: str, policy: Policy, seeds: range) -> dict[str, np.ndarray]:
    env = make_env(stage)
    observations, wins, values, seed_ids, steps, seconds = [], [], [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        start = len(observations)
        done = False
        step = 0
        reward = 0.0
        while not done:
            mask = env.action_masks()
            tick = perf_counter()
            action = policy.choose_action(obs, mask)
            seconds.append(perf_counter() - tick)
            diagnostics = getattr(policy, "last_diagnostics", None)
            values.append(float(getattr(diagnostics, "value", np.nan)) if diagnostics is not None else np.nan)
            observations.append(obs.copy())
            seed_ids.append(seed)
            steps.append(step)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step += 1
        wins.extend([int(terminated and reward > 0)] * (len(observations) - start))
    return dict(
        observations=np.asarray(observations, dtype=np.float32),
        win=np.asarray(wins, dtype=np.int8),
        search_value=np.asarray(values, dtype=np.float32),
        seed=np.asarray(seed_ids, dtype=np.int32),
        step=np.asarray(steps, dtype=np.int16),
        seconds=np.asarray(seconds, dtype=np.float32),
    )


def load_data(paths: list[str]) -> dict[str, np.ndarray]:
    parts = [np.load(path) for path in paths]
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files}


def lambda_returns(search: np.ndarray, outcome: np.ndarray, seeds: np.ndarray, lam: float) -> np.ndarray:
    """Per episode, G_t = (1 - lam) * root_t + lam * G_{t+1}, with G_T = outcome.

    `root_t` is the search root value at decision t (a one-turn backup of the
    leaf value); lam = 0 gives pure search targets and lam = 1 pure outcomes.
    Decisions without a search value fall back to the outcome.
    """
    targets = np.empty_like(search)
    following = None
    for index in range(len(search) - 1, -1, -1):
        if index == len(search) - 1 or seeds[index] != seeds[index + 1]:
            following = outcome[index]
        root = search[index] if np.isfinite(search[index]) else outcome[index]
        following = (1 - lam) * root + lam * following
        targets[index] = following
    return targets


def targets_for(data: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """'outcome', 'search' (root values), 'mix:<w>' (w*search + (1-w)*outcome),
    or 'lambda:<lam>' (per-episode lambda-returns over search root values)."""
    outcome = data["win"].astype(np.float64)
    if mode == "outcome":
        return outcome
    raw = data["search_value"].astype(np.float64)
    search = np.where(np.isfinite(raw), raw, outcome)
    if mode == "search":
        return search
    if mode.startswith("mix:"):
        weight = float(mode.split(":", 1)[1])
        return weight * search + (1 - weight) * outcome
    if mode.startswith("lambda:"):
        return lambda_returns(raw, outcome, data["seed"], float(mode.split(":", 1)[1]))
    raise ValueError(f"Unknown target mode {mode!r}")


def fit(stage: str, train: dict, val: dict, *, target: str, hidden: list[int], epochs: int,
        batch_size: int, lr: float, weight_decay: float, seed: int, threads: int) -> tuple[list, dict]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    layout = make_layout(stage)
    x_train = torch.tensor(value_features(layout, train["observations"]), dtype=torch.float32)
    y_train = torch.tensor(targets_for(train, target), dtype=torch.float32)
    x_val = torch.tensor(value_features(layout, val["observations"]), dtype=torch.float32)
    y_val_outcome = torch.tensor(val["win"].astype(np.float64), dtype=torch.float32)
    y_val_target = torch.tensor(targets_for(val, target), dtype=torch.float32)
    sizes = [x_train.shape[1], *hidden, 1]
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[index], sizes[index + 1]))
        if index < len(sizes) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()
    best = (float("inf"), None, -1)
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(len(x_train), generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(net(x_train[batch]).squeeze(1), y_train[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        scheduler.step()
        net.eval()
        with torch.no_grad():
            logits = net(x_val).squeeze(1)
            val_outcome = float(loss_fn(logits, y_val_outcome))
            val_target = float(loss_fn(logits, y_val_target))
            prob = torch.sigmoid(logits)
            brier = float(((prob - y_val_outcome) ** 2).mean())
        history.append(dict(epoch=epoch, train_loss=total / len(order), val_bce_outcome=val_outcome,
                            val_bce_target=val_target, val_brier_outcome=brier))
        print(json.dumps(history[-1]), flush=True)
        if val_target < best[0]:
            state = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                     for m in net if isinstance(m, nn.Linear)]
            best = (val_target, state, epoch)
    return best[1], dict(history=history, best_epoch=best[2], best_val_bce_target=best[0])


def fit_race(stage: str, data: dict, iterations: int = 50) -> tuple[float, float]:
    """Two-parameter logistic calibration of the race margin to episode outcomes."""
    margin = RaceValue(stage).margin(data["observations"])
    y = data["win"].astype(np.float64)
    x = np.stack([margin, np.ones_like(margin)], axis=1)
    theta = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-x @ theta))
        gradient = x.T @ (y - p)
        hessian = (x * (p * (1 - p))[:, None]).T @ x
        theta += np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
    return float(theta[0]), float(theta[1])


def save_value(path: str, weights: list, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, (w, b) in enumerate(weights):
        arrays[f"w{index}"] = w
        arrays[f"b{index}"] = b
    np.savez(path, meta=json.dumps(meta), **arrays)


def main() -> None:
    parser = ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect")
    c.add_argument("--stage", default="m4")
    c.add_argument("--policy", required=True, choices=["heuristic", "threat", "ppo", "expectimax"])
    c.add_argument("--model")
    c.add_argument("--leaf")
    c.add_argument("--depth", type=int, default=1)
    c.add_argument("--node-budget", type=int, default=None)
    c.add_argument("--epsilon", type=float, default=0.0)
    c.add_argument("--agent-seed", type=int, default=0)
    c.add_argument("--seed-start", type=int, required=True)
    c.add_argument("--episodes", type=int, required=True)
    c.add_argument("--out", required=True)
    f = commands.add_parser("fit")
    f.add_argument("--stage", default="m4")
    f.add_argument("--train", nargs="+", required=True)
    f.add_argument("--val", nargs="+", required=True)
    f.add_argument("--target", default="outcome")
    f.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    f.add_argument("--epochs", type=int, default=30)
    f.add_argument("--batch-size", type=int, default=2048)
    f.add_argument("--lr", type=float, default=2e-3)
    f.add_argument("--weight-decay", type=float, default=1e-4)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--threads", type=int, default=4)
    f.add_argument("--out", required=True)
    r = commands.add_parser("fit-race")
    r.add_argument("--stage", default="m4")
    r.add_argument("--train", nargs="+", required=True)
    r.add_argument("--val", nargs="+", required=True)
    args = parser.parse_args()

    if args.command == "collect":
        policy = make_policy(args.stage, args.policy, model=args.model, leaf=args.leaf, depth=args.depth,
                             always_search=True, node_budget=args.node_budget)
        if args.epsilon > 0:
            policy = EpsilonPolicy(policy, args.epsilon, args.agent_seed)
        tick = perf_counter()
        data = collect(args.stage, policy, range(args.seed_start, args.seed_start + args.episodes))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
        episodes_won = data["win"][np.r_[True, data["seed"][1:] != data["seed"][:-1]]].mean()
        print(json.dumps(dict(out=args.out, decisions=len(data["win"]), win_rate=float(episodes_won),
                              seconds=perf_counter() - tick)))
    elif args.command == "fit-race":
        scale, bias = fit_race(args.stage, load_data(args.train))
        val = load_data(args.val)
        prob = RaceValue(args.stage, scale=scale, bias=bias)(val["observations"])
        y = val["win"].astype(np.float64)
        eps = 1e-12
        bce = float(-np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))
        print(json.dumps(dict(scale=scale, bias=bias, val_bce_outcome=bce, val_brier_outcome=float(np.mean((prob - y) ** 2)))))
    else:
        train = load_data(args.train)
        val = load_data(args.val)
        weights, info = fit(args.stage, train, val, target=args.target, hidden=args.hidden, epochs=args.epochs,
                            batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
                            seed=args.seed, threads=args.threads)
        meta = dict(stage=args.stage, layers=len(weights), canonicalize=True, feature_set="obs", hidden=args.hidden,
                    target=args.target, train=args.train, val=args.val, train_samples=int(len(train["win"])),
                    **{k: v for k, v in info.items() if k != "history"})
        save_value(args.out, weights, meta)
        Path(args.out).with_suffix(".history.json").write_text(json.dumps(info, indent=2))
        print(json.dumps({k: v for k, v in meta.items() if k not in ("train", "val")}))


if __name__ == "__main__":
    main()
