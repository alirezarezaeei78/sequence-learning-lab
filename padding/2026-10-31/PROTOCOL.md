# Declared educational example — 31 October 2026

One objective: distinguish padding, truncation and a loss mask. Demonstrate that
zero input is not an LSTM stop instruction. This is authored synthetic arithmetic
with hand-set UNTRAINED weights; no dataset, fitted classifier, benchmark accuracy,
runtime comparison, scientific novelty or deployment claim.

Use PyTorch2.4.0 CPU float64, batch-first, one layer, one direction, input/hidden
width1, no dropout/projection. All input/recurrent weights0.5; input biases0.1;
recurrent biases0. Initial h=c=0. Two ordered sequences: A=[1,2], B=[0,1,-1,2].
B's first zero is real data. Right-pad to T4 with zeros; lengths[2,4], intentionally
unsorted. Compare ordinary last h_n, ordinary output at length-1, packed h_n and
each unpadded sequence's final h/c. Pack with enforce_sorted=False and CPU lengths.

Independent scalar math: z=.5*x+.5*h+.1; i=f=o=sigmoid(z), g=tanh(z),
c=f*c+i*g, h=o*tanh(c). Check all ordinary timesteps and standalone final h/c.
Change padded suffix from0to9: valid outputs of this forward-only model and packed
final states should agree, ordinary final state can change. No empirical winner
assumed. Show masked squared-output loss with correct valid-step denominator6
versus padded denominator8; mask only changes error reduction, not forward states.
Keep B's valid0 in mask; do not infer lengths from nonzero counts.

Truncation: explicitly retain first2 entries of B, update length to2; compare its
final state with full4-step B. Packing length1 for A instead of2 deliberately
demonstrates silent information removal from incorrect lengths. Validator rejects
zero/negative/non-integer/oversize lengths and count/shape mismatches rather than
depending on framework errors. Packing assumes valid prefixes/right padding;
left padding requires a different preprocessing contract.

Additional targeted checks: packed batch order restored after internal sorting;
all-equal lengths; extra padding cannot alter packed output; gradients of packed
final-state loss w.r.t padded inputs are0. Record complete computed values, actual
stdout/environment and source hashes. No random seed needed because every used
weight/input is explicitly assigned. Original static comparison infographic with
authored source, values and complete alt description. Test/inspect before GitHub.
No LinkedIn publication in this preparation; late window/native access controls
remain. Existing historical posts/media and uncommitted files remain untouched.
