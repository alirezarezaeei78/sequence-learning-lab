"""Authored float64 GRU step following PyTorch's reset-after convention."""
from __future__ import annotations

import numpy as np
from lstm_cell import sigmoid


def mix_state(candidate, previous, update):
    """PyTorch convention: z=1 keeps previous, z=0 selects candidate."""
    n, h, z = (np.asarray(v, dtype=np.float64) for v in
               (candidate, previous, update))
    if n.shape != h.shape or z.shape != h.shape:
        raise ValueError("candidate, previous and update shapes must match")
    if not all(np.isfinite(v).all() for v in (n, h, z)):
        raise ValueError("finite values required")
    if np.any((z < 0) | (z > 1)):
        raise ValueError("update gate must lie in [0,1]")
    return (1-z)*n + z*h


def step(x, h, weight_ih, weight_hh, bias_ih, bias_hh):
    """Return (h_next, trace); x:(B,D), h:(B,H), packed order r,z,n.

    Reset multiplies (h @ W_hn.T + b_hn), not h before the matrix multiply.
    This placement matters for non-diagonal matrices and nonzero biases.
    """
    x, h, wi, wh, bi, bh = (np.asarray(v, dtype=np.float64) for v in
                            (x, h, weight_ih, weight_hh, bias_ih, bias_hh))
    if not all(np.isfinite(v).all() for v in (x, h, wi, wh, bi, bh)):
        raise ValueError("all inputs and parameters must be finite")
    if x.ndim != 2 or h.ndim != 2:
        raise ValueError("x and h must be matrices")
    batch, features = x.shape
    hidden = h.shape[1]
    if min(batch, features, hidden) < 1 or h.shape[0] != batch:
        raise ValueError("nonempty matching dimensions required")
    if (wi.shape != (3*hidden, features) or wh.shape != (3*hidden, hidden)
            or bi.shape != (3*hidden,) or bh.shape != (3*hidden,)):
        raise ValueError("parameters must match r,z,n packing")
    with np.errstate(over="raise", invalid="raise"):
        ir, iz, inn = np.split(x @ wi.T + bi, 3, axis=1)
        hr, hz, hn = np.split(h @ wh.T + bh, 3, axis=1)
        r, z = sigmoid(ir+hr), sigmoid(iz+hz)
        n = np.tanh(inn + r*hn)
        out = mix_state(n, h, z)
    if not np.isfinite(out).all():
        raise ValueError("nonfinite output produced")
    return out, dict(r=r, z=z, n=n, retained=z*h, written=(1-z)*n)


def parameter_count(kind, input_size, hidden_size, bias=True):
    """Single layer/direction, unprojected PyTorch module, excluding a task head."""
    if kind not in ("lstm", "gru"):
        raise ValueError("kind must be lstm or gru")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
           for v in (input_size, hidden_size)):
        raise ValueError("positive integer sizes required")
    gates = 4 if kind == "lstm" else 3
    return gates*hidden_size*(input_size+hidden_size+(2 if bias else 0))


def worked_parameters():
    """Construct illustrative gates r=.25,z=.8,n=-.25 using nonzero weights."""
    x, h = np.array([[1.]]), np.array([[.5]])
    wi = np.array([[.2], [-.1], [.3]])
    wh = np.array([[.4], [.2], [.6]])
    bh = np.array([.05, -.1, .1])
    target = np.array([np.log(.25/.75), np.log(.8/.2), np.arctanh(-.25)])
    bi = np.array([target[0]-.2-.4*.5-.05,
                   target[1]+.1-.2*.5+.1,
                   target[2]-.3-.25*(.6*.5+.1)])
    return x, h, wi, wh, bi, bh
