# LSTM versus GRU: checked teaching protocol

Declared 24 October 2026 before running the new example. This is an educational
comparison of standard unprojected LSTM and PyTorch-style GRU state updates.
It is not model training, a sensor benchmark, a runtime comparison, or new research.

- Question: which state is carried, what do the gates mix, and how many parameters
  does a fixed single-layer recurrent module contain?
- Authored inputs only: deterministic float64 fixtures, seed 24. No data download,
  personal data, fitted task, held-out split, or prediction-performance claim.
- Primary arithmetic checks: NumPy equations versus installed PyTorch 2.4.0 CPU
  LSTMCell/GRUCell on matching weights and inputs; independently computed parameter
  formulas versus actual module tensors. Preserve every result and mismatch.
- Configuration: input width D=9, hidden width H=16, one layer, one direction,
  biases enabled, LSTM projection disabled. Count recurrent parameters only,
  excluding a task head. PyTorch stores two bias vectors per gate block.
- Counts: LSTM 4H(D+H+2); GRU 3H(D+H+2). Verify by execution before publication.
- GRU convention: r,z,n packing; reset multiplies the recurrent candidate affine
  result, including its recurrent bias. The update uses (1-z)*n + z*h_previous.
  Other formulations use another reset placement or gate convention.
- Worked GRU fixture: h_previous=0.50, z=0.80, n=-0.25, so retained=0.40,
  new candidate contribution=-0.05 and next h=0.35. Construct nonzero parameters
  to obtain the gates, then check the framework result.
- Failure checks: malformed/nonfinite inputs, extreme sigmoid inputs, independent
  batch rows, seeded multistep parity, noncommuting reset-placement counterexample,
  update-gate endpoints, and batch-first sequence versus final-state shapes.
- Budget: seconds of CPU arithmetic, no optimizer/training/GPU or timing claim.
- Visual: original two-column comparison and parameter-count table, plus an
  optional staged GIF highlighting a schematic GRU update. Duration is presentation
  pacing, never execution speed. Preserve SVG, PNG, GIF and the rendering script.
- Source basis: official tagged PyTorch 2.4.0 module documentation/source.
  Prior 20 October tutorial explained only an LSTM step; this adds GRU reset/update
  behavior, a reset-placement counterexample, framework parity and fixed-width counts.
- Publication: prepared after the daily window closed. No automatic late LinkedIn
  submission, retroactive random time or backfill. Save as unpublished material.
