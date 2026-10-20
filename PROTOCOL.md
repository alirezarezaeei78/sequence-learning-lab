# LSTM gate arithmetic: frozen tutorial protocol

Declared 20 October 2026, before executing the example. This is an established-method teaching example, not a trained activity classifier or model comparison.

## Learning question

How do a standard, unprojected LSTM's gates transform the previous cell state into the next cell state and hidden output?

## Scope and intervention

Implement the documented batched LSTMCell equations in NumPy, with separate input and recurrent weights and biases. Use gate order **input, forget, candidate, output (i, f, g, o)**. State and output shapes are `(B, H)`; one timestep's input is `(B, D)`. The sequence runner accepts `(B, T, D)` and iterates over T. No projection, peepholes, dropout, masking or bidirectionality is implemented.

The main scalar illustration has x=1, previous h=0.2, previous c=1.5, with authored affine parameters chosen to yield i=0.25, f=0.8, g=-0.4 and o=0.6. These are **illustrative, deliberately constructed values**, not trained gates. Expected c is 0.8*1.5+0.25*(-0.4)=1.1; expected h is 0.6*tanh(1.1). A second controlled illustration repeats the memory-only path with fixed f in {0.5, 0.8, 0.99} and zero candidate. It demonstrates multiplication, not learned retention, model quality or gradient guarantees.

## Checks specified in advance

- Compare the vectorized implementation with a separately written scalar-loop oracle, using deterministic synthetic cases with several batch/input/hidden shapes and seed 13.
- Check the main scalar gates and states against hand-specified arithmetic at float64 tolerances.
- Compare sequence execution with scalar-loop execution at every timestep; verify batch independence, output shapes, repeatability and no mutation of inputs.
- Verify shape/finite-value validation and stable sigmoid at extreme finite logits.
- Check that two biases are added, changing gate order changes the constructed answer, and c need not lie in [-1,1] while h does for this unprojected cell.
- Check fixed-gate memory multiplication against f**t, including a finite-difference partial derivative of c with respect to previous c **holding gates fixed**. This is not the full recurrent-network gradient.
- Save all executed illustrative inputs/outputs, checks, versions and commands. No data split or predictive metric is appropriate: no model is fitted, and no real-world dataset is used.

## Evidence and media

Generate a static original infographic, editable SVG and a 20-second silent animation from the saved numerical results. Stages reveal existing values; animation duration is not computation time. Inspect first/middle/last frames, spelling, numeric values, contrast, legibility and actual playback. Preserve a complete static version if native video upload cannot be verified.

## Sources, data and budget

Primary reference: PyTorch 2.9 LSTMCell documentation and its tagged official source; consulted 20 October 2026. Source ledger records exact URLs and access hashes. No third-party code or illustrations are copied. Authored synthetic inputs and code are released under MIT. No external dataset, customer data, training, held-out evaluation, business-impact or scientific-novelty claim. Budget: CPU, under 5 minutes for checks and under 10 minutes for rendering on existing hardware; no paid compute.
