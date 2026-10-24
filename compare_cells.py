"""Execute a reproducible LSTM/GRU teaching example, without training."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
from gru_cell import parameter_count, step as gru_step, worked_parameters
from lstm_cell import step as lstm_step


def framework_checks():
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(24)
    rng = np.random.default_rng(24)
    cases = []
    for kind, gates in (("lstm", 4), ("gru", 3)):
        for batch, features, hidden in ((1, 1, 1), (3, 2, 4), (2, 9, 16)):
            cls = torch.nn.LSTMCell if kind == "lstm" else torch.nn.GRUCell
            cell = cls(features, hidden, dtype=torch.float64)
            arrays = [rng.normal(0, .3, size=s) for s in
                      ((gates*hidden, features), (gates*hidden, hidden),
                       (gates*hidden,), (gates*hidden,))]
            with torch.no_grad():
                for param, value in zip(cell.parameters(), arrays):
                    param.copy_(torch.from_numpy(value))
            h = rng.normal(size=(batch, hidden)); c = rng.normal(size=h.shape)
            ht, ct = torch.from_numpy(h.copy()), torch.from_numpy(c.copy())
            errors = []
            for _ in range(5):
                x = rng.normal(size=(batch, features))
                with torch.no_grad():
                    if kind == "lstm":
                        h, c, _ = lstm_step(x, h, c, *arrays)
                        ht, ct = cell(torch.from_numpy(x), (ht, ct))
                        errors.append(float(np.max(np.abs(c-ct.numpy()))))
                    else:
                        h, _ = gru_step(x, h, *arrays)
                        ht = cell(torch.from_numpy(x), ht)
                errors.append(float(np.max(np.abs(h-ht.numpy()))))
            assert max(errors) < 1e-12
            cases.append(dict(kind=kind, batch=batch, input_size=features,
                              hidden_size=hidden, steps=5,
                              max_abs_error=max(errors)))
    counts = {}
    shapes = {}
    for kind, cls in (("lstm", torch.nn.LSTM), ("gru", torch.nn.GRU)):
        for bias in (True, False):
            module = cls(9, 16, batch_first=True, bias=bias, dtype=torch.float64)
            actual = sum(p.numel() for p in module.parameters())
            assert actual == parameter_count(kind, 9, 16, bias)
            counts[f'{kind}_bias_{bias}'] = actual
        module = cls(9, 16, batch_first=True, dtype=torch.float64)
        output, state = module(torch.zeros((2, 5, 9), dtype=torch.float64))
        states = list(state) if isinstance(state, tuple) else [state]
        shapes[kind] = dict(output=list(output.shape),
                            states=[list(v.shape) for v in states])
    x,h,*parameters=worked_parameters()
    cell=torch.nn.GRUCell(1,1,dtype=torch.float64)
    with torch.no_grad():
        for param,value in zip(cell.parameters(),parameters):param.copy_(torch.from_numpy(value))
        worked=float(cell(torch.from_numpy(x),torch.from_numpy(h))[0,0])
    assert abs(worked-.35)<1e-14
    return dict(version=torch.__version__, device="CPU", dtype="float64",
                worked_gru_next=worked,
                parity_cases=cases, actual_parameter_counts=counts, shapes=shapes)


def run():
    out, trace = gru_step(*worked_parameters())
    assert np.allclose(out, .35, rtol=0, atol=1e-14)
    return dict(scope="Authored synthetic teaching; no training or benchmark",
                seed=24, versions=dict(python=platform.python_version(), numpy=np.__version__),
                hardware=dict(os=platform.system(), processor=platform.processor()),
                configuration=dict(input_size=9, hidden_size=16, layers=1,
                                   bidirectional=False, bias=True, projection=0),
                worked_gru=dict(previous=.5, **{k:float(v[0,0]) for k,v in trace.items()},
                                next=float(out[0,0])),
                parameter_counts={k:parameter_count(k,9,16) for k in ("lstm","gru")},
                framework=framework_checks(),
                limitation="Counts and arithmetic do not establish speed, accuracy or memory-horizon superiority.")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='comparisons/2026-10-24/results.json')
    target=Path(parser.parse_args().output);result=run();target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
