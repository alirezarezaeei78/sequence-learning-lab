"""Inspectable standard LSTM cell arithmetic. No training or framework dependency."""
from __future__ import annotations

import numpy as np


def sigmoid(value):
    """Stable float64 sigmoid for finite inputs."""
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError("sigmoid input must be finite")
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def step(x, h, c, weight_ih, weight_hh, bias_ih, bias_hh):
    """Return (h_next, c_next, gate_trace), all with explicit batch dimensions.

    x: (B,D); h,c: (B,H); weight_ih: (4H,D); weight_hh: (4H,H);
    both biases: (4H,). Packed gate order is i,f,g,o.
    """
    values = [np.asarray(v, dtype=np.float64) for v in
              (x, h, c, weight_ih, weight_hh, bias_ih, bias_hh)]
    if not all(np.isfinite(v).all() for v in values):
        raise ValueError("all inputs and parameters must be finite")
    x, h, c, wi, wh, bi, bh = values
    if x.ndim != 2 or h.ndim != 2 or c.shape != h.shape:
        raise ValueError("x and h must be matrices; c must match h")
    batch, features = x.shape
    if batch < 1 or features < 1 or h.shape[1] < 1 or h.shape[0] != batch:
        raise ValueError("nonempty matching batch/input/hidden dimensions required")
    hidden = h.shape[1]
    if (wi.shape != (4*hidden, features) or wh.shape != (4*hidden, hidden)
            or bi.shape != (4*hidden,) or bh.shape != (4*hidden,)):
        raise ValueError("parameter shapes must match D, H and i,f,g,o packing")
    with np.errstate(over="raise", invalid="raise"):
        pre = x @ wi.T + bi + h @ wh.T + bh
        zi, zf, zg, zo = np.split(pre, 4, axis=1)
        i, f, g, o = sigmoid(zi), sigmoid(zf), np.tanh(zg), sigmoid(zo)
        retained, written = f*c, i*g
        cn = retained + written
        hn = o*np.tanh(cn)
    if not np.isfinite(hn).all() or not np.isfinite(cn).all():
        raise ValueError("nonfinite state produced")
    return hn, cn, dict(i=i, f=f, g=g, o=o, retained=retained, written=written)


def unroll(sequence, h, c, *parameters):
    """Run a nonempty batch-first (B,T,D) sequence, returning every h and c."""
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 3 or sequence.shape[1] < 1:
        raise ValueError("sequence must have shape (B,T,D) with T >= 1")
    hs, cs = [], []
    for time in range(sequence.shape[1]):
        h, c, _ = step(sequence[:, time, :], h, c, *parameters)
        hs.append(h)
        cs.append(c)
    return np.stack(hs, axis=1), np.stack(cs, axis=1)


def worked_parameters():
    """Nonzero authored weights; biases construct the stated illustrative gates."""
    x, h, c = np.array([[1.0]]), np.array([[0.2]]), np.array([[1.5]])
    wi = np.array([[0.3], [-0.2], [0.4], [0.1]])
    wh = np.array([[0.1], [0.2], [-0.3], [0.4]])
    target = np.array([np.log(.25/.75), np.log(.8/.2), np.arctanh(-.4),
                       np.log(.6/.4)])
    bh = np.array([.05, -.1, .15, -.2])
    bi = target - (wi @ x[0] + wh @ h[0]) - bh
    return x, h, c, wi, wh, bi, bh
