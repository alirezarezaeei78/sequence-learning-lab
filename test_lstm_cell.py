"""Independent loop oracle and arithmetic/shape/edge checks, not a benchmark."""
import copy
import math
import unittest

import numpy as np

from lstm_cell import sigmoid, step, unroll, worked_parameters


def scalar_oracle(x, h, c, wi, wh, bi, bh):
    """Independent Python loops and math functions, no vectorized cell reuse."""
    batch, hidden = h.shape
    hn, cn = np.zeros_like(h), np.zeros_like(c)
    for b in range(batch):
        for j in range(hidden):
            gates = []
            for gate in range(4):
                row = gate*hidden+j
                z = float(bi[row])+float(bh[row])
                for k in range(x.shape[1]):
                    z += float(wi[row, k])*float(x[b, k])
                for k in range(hidden):
                    z += float(wh[row, k])*float(h[b, k])
                gates.append(math.tanh(z) if gate == 2 else 1/(1+math.exp(-z)))
            i, f, g, o = gates
            cn[b, j] = f*c[b, j] + i*g
            hn[b, j] = o*math.tanh(cn[b, j])
    return hn, cn


def fixture(batch=2, features=3, hidden=4):
    rng = np.random.default_rng(13)
    shapes = [(batch, features), (batch, hidden), (batch, hidden),
              (4*hidden, features), (4*hidden, hidden), (4*hidden,), (4*hidden,)]
    return [rng.normal(0, .4, shape) for shape in shapes]


class CellChecks(unittest.TestCase):
    def test_worked_gates(self):
        _, _, trace = step(*worked_parameters())
        np.testing.assert_allclose([trace[k].item() for k in 'ifgo'], [.25,.8,-.4,.6], atol=1e-14)

    def test_hand_arithmetic(self):
        h, c, trace = step(*worked_parameters())
        self.assertAlmostEqual(trace['retained'].item(), 1.2, places=14)
        self.assertAlmostEqual(trace['written'].item(), -.1, places=14)
        self.assertAlmostEqual(c.item(), 1.1, places=14)
        self.assertAlmostEqual(h.item(), .6*math.tanh(1.1), places=14)

    def test_independent_oracle(self):
        for dims in [(1,1,1),(2,3,4),(3,2,2),(1,7,3)]:
            values=fixture(*dims); h,c,_=step(*values); oh,oc=scalar_oracle(*values)
            np.testing.assert_allclose(h,oh,rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(c,oc,rtol=1e-13,atol=1e-14)

    def test_sequence_oracle(self):
        values=fixture(); sequence=np.random.default_rng(29).normal(size=(2,5,3))
        hs,cs=unroll(sequence,*values[1:]); h,c=values[1:3]
        self.assertEqual(hs.shape,(2,5,4))
        for t in range(5):
            h,c=scalar_oracle(sequence[:,t,:],h,c,*values[3:])
            np.testing.assert_allclose(hs[:,t,:],h,atol=1e-14)
            np.testing.assert_allclose(cs[:,t,:],c,atol=1e-14)

    def test_batch_independence(self):
        values=fixture(); h,c,_=step(*values)
        for b in range(2):
            one=[v[b:b+1] if j<3 else v for j,v in enumerate(values)]
            oh,oc,_=step(*one)
            np.testing.assert_allclose(oh,h[b:b+1],atol=1e-14)
            np.testing.assert_allclose(oc,c[b:b+1],atol=1e-14)

    def test_repeat_and_no_mutation(self):
        values=fixture(); before=copy.deepcopy(values)
        a=step(*values); b=step(*values)
        for left,right in zip(a[:2],b[:2]):np.testing.assert_array_equal(left,right)
        for left,right in zip(values,before):np.testing.assert_array_equal(left,right)

    def test_stable_sigmoid_extremes(self):
        with np.errstate(over='raise'):
            values=sigmoid(np.array([-1000.,0.,1000.]))
        np.testing.assert_array_equal(values,[0.,.5,1.])

    def test_two_bias_terms(self):
        values=fixture(); a=step(*values)
        combined=copy.deepcopy(values); combined[-2]+=combined[-1]; combined[-1][:]=0
        b=step(*combined)
        np.testing.assert_allclose(a[0],b[0],atol=1e-14)

    def test_wrong_packing_changes_answer(self):
        values=list(worked_parameters()); right=step(*values)
        for j in range(3,7):values[j]=values[j][[1,0,2,3]]
        wrong=step(*values)
        self.assertGreater(abs(wrong[1].item()-right[1].item()),.1)

    def test_cell_not_tanh_bounded_hidden_is(self):
        h,c,_=step(*worked_parameters())
        self.assertGreater(c.item(),1)
        self.assertLessEqual(abs(h.item()),1)

    def test_shape_rejections(self):
        for index, replacement in [(0,np.zeros(3)),(1,np.zeros((3,4))),
              (2,np.zeros((2,3))),(3,np.zeros((15,3))),(4,np.zeros((16,5))),
              (5,np.zeros(15)),(6,np.zeros((16,1)))]:
            values=fixture();values[index]=replacement
            with self.assertRaises(ValueError):step(*values)

    def test_finite_rejections(self):
        for index in range(7):
            values=fixture();values[index].flat[0]=np.nan
            with self.assertRaises(ValueError):step(*values)
        with self.assertRaises(ValueError):sigmoid([np.inf])

    def test_empty_dimension_rejections(self):
        for dims in [(0,3,4),(2,0,4),(2,3,0)]:
            with self.assertRaises(ValueError):step(*fixture(*dims))
        with self.assertRaises(ValueError):unroll(np.empty((2,0,3)),*fixture()[1:])

    def test_memory_only_products(self):
        for f in [.5,.8,.99]:
            c=1.
            for t in range(1,11):
                c=f*c
                self.assertAlmostEqual(c,f**t,places=14)

    def test_fixed_gate_partial_derivative(self):
        # Holding all gates fixed; NOT the total derivative through h_previous.
        f,i,g=.8,.25,-.4; c=1.5; eps=1e-6
        derivative=((f*(c+eps)+i*g)-(f*(c-eps)+i*g))/(2*eps)
        self.assertAlmostEqual(derivative,f,places=8)


if __name__=='__main__':unittest.main(verbosity=2)
