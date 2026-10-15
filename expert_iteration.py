"""Collect explored expert states and fit value targets plus a masked policy prior.

`collect` always runs the teacher before optional exploration: search labels
belong to the recorded state even when the behavior action is random. `fit`
distills the conservative teacher and fits a value on search, TD(lambda), or
outcome targets. Repeat collect/fit with the previous checkpoints to iterate.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from agents.expectimax_agent import LeafValue
from agents.heuristic_agent import make_heuristic_agent
from agents.guided_planner import GuidedPlannerAgent
from agents.policy import Policy
from agents.policy_prior import MLPPolicyPrior
from agents.value_functions import make_layout, value_features
from combat.stages import make_env
from evaluation.planning_eval import make_leaf
from value_training import fit, load_data, save_value


def collect_expert(stage: str, teacher: Policy, seeds: range, *, epsilon: float = 0.1,
                   agent_seed: int = 0, leaf: LeafValue | None = None) -> dict[str, np.ndarray]:
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon must be in [0, 1]')
    env = make_env(stage)
    rows = {k: [] for k in ('observations', 'mask', 'action', 'expert_action', 'q_values',
                           'search_value', 'next_value', 'done', 'win', 'seed', 'step')}
    for seed in seeds:
        # Independent per-episode exploration is reproducible across chunking.
        rng = np.random.default_rng(np.random.SeedSequence([agent_seed, seed]))
        obs, _ = env.reset(seed=seed)
        start = len(rows['action'])
        step = 0
        while True:
            mask = env.action_masks()
            expert = teacher.choose_action(obs, mask)
            if expert not in np.flatnonzero(mask):
                raise ValueError('Teacher selected an illegal action')
            diagnostics = getattr(teacher, 'last_diagnostics', None)
            q = np.full(env.action_space.n, np.nan)
            if diagnostics is not None:
                for action, value in diagnostics.q_values:
                    q[action] = value
            selected_value = getattr(diagnostics, 'selected_value', np.nan)
            action = int(rng.choice(np.flatnonzero(mask))) if rng.random() < epsilon else expert
            following, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_value = float(terminated and reward > 0) if done else (
                float(np.asarray(leaf(following[None])).reshape(-1)[0]) if leaf is not None else np.nan)
            for key, value in dict(observations=obs.copy(), mask=mask.copy(), action=action,
                                   expert_action=expert, q_values=q, search_value=selected_value,
                                   next_value=next_value, done=done, seed=seed, step=step).items():
                rows[key].append(value)
            step += 1
            if done:
                rows['win'].extend([int(terminated and reward > 0)] * (len(rows['action']) - start))
                break
            obs = following
    env.close()
    dtypes = dict(observations=np.float32, mask=bool, action=np.int16, expert_action=np.int16,
                  q_values=np.float32, search_value=np.float32, next_value=np.float32,
                  done=bool, win=np.int8, seed=np.int32, step=np.int16)
    return {k: np.asarray(v, dtype=dtypes[k]) for k, v in rows.items()}


def fit_prior(stage: str, train: dict, val: dict, *, hidden: list[int], epochs: int = 20,
              batch_size: int = 512, lr: float = 0.002, seed: int = 0,
              threads: int = 2) -> tuple[list, dict]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    layout = make_layout(stage)

    def tensors(data):
        masks = np.asarray(data['mask'], dtype=bool)
        y = np.asarray(data['expert_action'], dtype=int)
        if len(y) == 0 or masks.ndim != 2 or np.any(~masks[np.arange(len(y)), y]):
            raise ValueError('Policy training needs legal expert actions and nonempty data')
        return (torch.tensor(value_features(layout, data['observations']), dtype=torch.float32),
                torch.tensor(masks), torch.tensor(y, dtype=torch.long))

    xt, mt, yt = tensors(train)
    xv, mv, yv = tensors(val)
    sizes = [xt.shape[1], *hidden, mt.shape[1]]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    net = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_loss, best_weights, best_epoch = float('inf'), None, -1
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(net(xt[batch]).masked_fill(~mt[batch], -torch.inf), yt[batch])
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = net(xv).masked_fill(~mv, -torch.inf)
            val_loss = float(loss_fn(logits, yv))
            accuracy = float((logits.argmax(dim=1) == yv).float().mean())
        history.append(dict(epoch=epoch, val_cross_entropy=val_loss, val_accuracy=accuracy))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_weights = [(layer.weight.detach().numpy().T.copy(), layer.bias.detach().numpy().copy())
                            for layer in net if isinstance(layer, nn.Linear)]
    if best_weights is None:
        raise ValueError('Need at least one fitting epoch')
    return best_weights, dict(best_epoch=best_epoch, best_val_cross_entropy=best_loss, history=history)


def check_seed_split(train: dict, val: dict) -> None:
    if set(train['seed']) & set(val['seed']):
        raise ValueError('Training and validation episode seeds overlap')
    if any(130000 <= int(s) <= 132999 for s in np.r_[train['seed'], val['seed']]):
        raise ValueError('Frozen M5/M6 test seeds cannot be used for fitting')


def main() -> None:
    parser = ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    c = commands.add_parser('collect')
    c.add_argument('--stage', required=True, choices=('m5', 'm6'))
    c.add_argument('--leaf', help='omit for heuristic bootstrap labels/outcomes')
    c.add_argument('--prior')
    c.add_argument('--horizon', type=int, default=1)
    c.add_argument('--root-top-k', type=int, default=0)
    c.add_argument('--interior-top-k', type=int, default=2)
    c.add_argument('--node-budget', type=int, default=64)
    c.add_argument('--deviation-margin', type=float, default=0.03)
    c.add_argument('--epsilon', type=float, default=0.1)
    c.add_argument('--agent-seed', type=int, default=0)
    c.add_argument('--seed-start', type=int, required=True)
    c.add_argument('--episodes', type=int, required=True)
    c.add_argument('--out', required=True)
    f = commands.add_parser('fit')
    f.add_argument('--stage', required=True, choices=('m5', 'm6'))
    f.add_argument('--train', nargs='+', required=True)
    f.add_argument('--val', nargs='+', required=True)
    f.add_argument('--target', default='search')
    f.add_argument('--hidden', type=int, nargs='+', default=[64, 64])
    f.add_argument('--epochs', type=int, default=20)
    f.add_argument('--seed', type=int, default=0)
    f.add_argument('--out', required=True, help='directory for value.npz and prior.npz')
    args = parser.parse_args()
    if args.command == 'collect':
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        seeds = range(args.seed_start, args.seed_start + args.episodes)
        if set(seeds) & set(range(130000, 133000)):
            parser.error('Frozen test seeds cannot be used for data collection')
        anchor = make_heuristic_agent(args.stage, 'tuned' if args.stage == 'm5' else 'adaptive')
        leaf = make_leaf(args.stage, args.leaf) if args.leaf else None
        prior = MLPPolicyPrior.load(args.prior) if args.prior else None
        if prior is not None and prior.layout.stage != args.stage:
            parser.error('Prior stage mismatch')
        if getattr(getattr(leaf, 'layout', None), 'stage', args.stage) != args.stage:
            parser.error('Value stage mismatch')
        if prior is not None and leaf is None:
            parser.error('--prior requires --leaf')
        teacher = GuidedPlannerAgent(args.stage, leaf, anchor, prior=prior, horizon=args.horizon,
                                     root_top_k=args.root_top_k, interior_top_k=args.interior_top_k,
                                     node_budget=args.node_budget, deviation_margin=args.deviation_margin) if leaf else anchor
        data = collect_expert(args.stage, teacher, seeds, epsilon=args.epsilon, agent_seed=args.agent_seed, leaf=leaf)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out, **data)
        metadata = vars(args) | dict(rows=len(data['win']), wins=int(data['win'][data['done']].sum()))
        Path(args.out).with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(json.dumps(metadata))
    else:
        if args.epochs < 1 or any(size < 1 for size in args.hidden):
            parser.error('epochs and hidden layer sizes must be positive')
        train, val = load_data(args.train), load_data(args.val)
        check_seed_split(train, val)
        if args.target != 'outcome' and (not np.all(np.isfinite(train['search_value'])) or not np.all(np.isfinite(val['search_value']))):
            parser.error('Search/TD fitting requires expert data; bootstrap with --target outcome')
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        weights, info = fit(args.stage, train, val, target=args.target, hidden=args.hidden, epochs=args.epochs,
                            batch_size=512, lr=0.002, weight_decay=1e-4, seed=args.seed, threads=2)
        meta = dict(stage=args.stage, layers=len(weights), canonicalize=True, target=args.target,
                    train=args.train, val=args.val, train_samples=len(train['win']), kind='value')
        save_value(str(out / 'value.npz'), weights, meta)
        prior_weights, prior_info = fit_prior(args.stage, train, val, hidden=args.hidden, epochs=args.epochs, seed=args.seed)
        save_value(str(out / 'prior.npz'), prior_weights, meta | dict(layers=len(prior_weights), kind='policy_prior'))
        (out / 'fit.json').write_text(json.dumps(dict(value=info, prior=prior_info, config=vars(args)), indent=2) + '\n')
        print(json.dumps(dict(out=str(out), prior_val_ce=prior_info['best_val_cross_entropy'], value_val_bce=info['best_val_bce_target'])))


if __name__ == '__main__':
    main()
