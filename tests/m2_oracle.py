"""Independent M2 value-iteration oracle (round limit ignored).

Condensed from the supervisor's exact solver
(runs/m2_exact_optimum_9000_13999/m2_dp.py): it shares no code with
`combat/`, so agreement with the exact transition model is a real check.
State V[f, e0, e1, sw, su, cl, a, b].
"""

import numpy as np

FMAX = 20
F_AC, F_ATK, F_DIE, F_BONUS = 16, 5, 8, 3


def dmg_dist(atk, ac, die, bonus, halve=False):
    dist = {}
    for roll in range(1, 21):
        crit = roll == 20
        if not (crit or (roll != 1 and roll + atk >= ac)):
            dist[0] = dist.get(0, 0) + 1 / 20
            continue
        sums = {0: 1.0}
        for _ in range(2 if crit else 1):
            sums = _roll(sums, die)
        for s, p in sums.items():
            dmg = max(0, s + bonus)
            if halve:
                dmg = max(1, dmg // 2)
            dist[dmg] = dist.get(dmg, 0) + p / 20
    return dist


def _roll(sums, die):
    out = {}
    for s, p in sums.items():
        for d in range(1, die + 1):
            out[s + d] = out.get(s + d, 0) + p / die
    return out


def _apply(V, axis, dist, n):
    out = np.zeros_like(V)
    ar = np.arange(n)
    for d, p in dist.items():
        out += p * np.take(V, np.maximum(ar - d, 0), axis=axis)
    return out


def _heal(V):
    out = np.zeros_like(V)
    ar = np.arange(FMAX + 1)
    for h in range(3, 13):
        out += 0.1 * np.take(V, np.minimum(ar + h, FMAX), axis=0)
    return out


def solve(enemies, tol=1e-12, max_iter=5000):
    """Return optimal V and Q (Q[action] with -1 for illegal) for one encounter."""
    H0, H1 = enemies[0]["hp"], enemies[1]["hp"]
    shape = (FMAX + 1, H0 + 1, H1 + 1, 2, 2, 2, 3, 2)
    fd = [dmg_dist(F_ATK, e["ac"], F_DIE, F_BONUS) for e in enemies]
    fh = [dmg_dist(F_ATK, e["ac"], F_DIE, F_BONUS, halve=True) for e in enemies]
    ed = [dmg_dist(e["atk"], F_AC, e["die"], e["bonus"]) for e in enemies]
    f_idx = np.arange(FMAX + 1).reshape(-1, 1, 1, 1, 1, 1, 1, 1)
    alive0 = np.arange(H0 + 1).reshape(1, -1, 1, 1, 1, 1, 1, 1) > 0
    alive1 = np.arange(H1 + 1).reshape(1, 1, -1, 1, 1, 1, 1, 1) > 0
    win = np.broadcast_to((f_idx > 0) & ~alive0 & ~alive1, shape)
    terminal = win | np.broadcast_to(f_idx == 0, shape)
    V = np.where(win, 1.0, 0.0)
    for _ in range(max_iter):
        Q = np.full((6,) + shape, -1.0)
        for i, axis in ((0, 1), (1, 2)):
            Q[i][..., 1:3, :] = _apply(V[..., 0:2, :], axis, fd[i], shape[axis])
            Q[i] = np.where(np.broadcast_to(alive0 if i == 0 else alive1, shape), Q[i], -1.0)
        nxt = _apply(V[:, :, :, :, :, 0, 0:2, :], 1, fh[0], H0 + 1)
        Q[2][:, :, :, :, :, 1, 1:3, :] = _apply(nxt, 2, fh[1], H1 + 1)
        Q[3][:, :, :, 1, :, :, :, 1] = _heal(V[:, :, :, 0, :, :, :, 0])
        Q[3][FMAX] = -1.0
        Q[4][:, :, :, :, 1, :, 0:2, :] = V[:, :, :, :, 0, :, 1:3, :]
        T = V[..., 1, 1]
        W1 = np.where(alive1[..., 0, 0], _apply(T, 0, ed[1], FMAX + 1), T)
        W0 = np.where(alive0[..., 0, 0], _apply(W1, 0, ed[0], FMAX + 1), W1)
        Q[5] = W0[..., None, None]
        Vn = np.where(terminal, V, Q.max(axis=0))
        delta = np.abs(Vn - V).max()
        V = Vn
        if delta < tol:
            break
    return V, Q
