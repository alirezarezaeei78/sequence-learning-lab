# LSTM versus GRU: inspect states, gates and counts

This original educational comparison adds a GRU step and a fixed-width parameter
audit to the earlier LSTM-only example. All fixtures are authored, synthetic and
untrained. No task dataset, accuracy result or runtime comparison is implied.

From the repository root, with Python 3.12:

```sh
python -m pip install -r requirements-comparison.txt
python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
python -m unittest -v
python compare_cells.py
python render_comparison.py
```

The new renderer creates PNG/SVG and a 15-second animated GIF image only. It
never creates or converts to a video file. The static image is complete without
motion. The loop highlights the retain, candidate and combine terms; it does not
animate learned behavior or measure execution time. Native LinkedIn playback
has not been verified; use the static image if GIF support is uncertain.

## What to try

In `gru_cell.worked_parameters`, the authored gates produce a hidden update of
0.35: retain 0.80 × 0.50 = 0.40 and add 0.20 × -0.25 = -0.05. Under the documented
PyTorch convention, an update gate near one keeps more of the previous state.
Change the previous state and inspect the trace. Other conventions may name the
complementary gate as the update gate, so read the actual equation.

The reset acts on the recurrent candidate affine result, including its recurrent
bias. `test_reset_placement_is_not_interchangeable` demonstrates why moving it
before a non-diagonal matrix multiplication changes the result.

For D=9 and H=16, a single-layer, one-direction PyTorch module with biases has
1,728 LSTM or 1,296 GRU parameters. These exclude a task head, projections and
extra layers. The formulas count both input and recurrent bias vectors. The
25% difference does not establish lower latency or better predictive quality.

With batch-first input `(2,5,9)`, both module outputs are `(2,5,16)`. Final states
are `(1,2,16)`: two tensors for LSTM and one for GRU. `batch_first` does not change
state-axis order. These are single-direction, one-layer examples only.

## Executed checks and limits

Saved `results.json` includes six matching NumPy/PyTorch cell comparisons across
five timesteps per configuration, both bias/no-bias parameter audits, output and
state shapes, dependency versions and the scalar worked result. Tests retain
the old LSTM independent oracle and add GRU convention/edge/failure cases.

No models were fitted, and there is no held-out predictive benchmark, training
speed result or memory-horizon guarantee. The GRU code is educational NumPy
arithmetic without autodiff, masking, projection, dropout or deployment features.
See `PROTOCOL.md` for declarations made before execution.

Sources: [official PyTorch 2.4 LSTM reference](https://docs.pytorch.org/docs/2.4/generated/torch.nn.LSTM.html),
[official tagged 2.4 implementation/documentation](https://github.com/pytorch/pytorch/blob/v2.4.0/torch/nn/modules/rnn.py),
and [current official GRU reference](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GRU.html).
The local framework check uses 2.4.0, not the current documentation version.
Only mathematical facts are paraphrased. No third-party art, model weights,
datasets or source code are redistributed. Original code, fixtures and media
are under the repository's MIT licence; external sources retain their licences.
