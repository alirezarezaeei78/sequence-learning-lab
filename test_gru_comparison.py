"""Meaningful GRU convention, arithmetic and framework-parity checks."""
import unittest
import importlib.util
import numpy as np
from gru_cell import step, mix_state, parameter_count, worked_parameters
from lstm_cell import sigmoid


class GRUChecks(unittest.TestCase):
    def test_worked_arithmetic(self):
        out, g=step(*worked_parameters())
        for name, value in dict(r=.25,z=.8,n=-.25,retained=.4,written=-.05).items():
            np.testing.assert_allclose(g[name],value,rtol=0,atol=1e-14)
        np.testing.assert_allclose(out,.35,rtol=0,atol=1e-14)

    def test_update_convention_endpoints(self):
        np.testing.assert_equal(mix_state(np.array([-.25]),np.array([.5]),np.ones(1)),[.5])
        np.testing.assert_equal(mix_state(np.array([-.25]),np.array([.5]),np.zeros(1)),[-.25])

    def test_reset_placement_is_not_interchangeable(self):
        h=np.array([[1.,2.]]);r=np.array([[.2,.8]])
        wh=np.array([[0.,2.],[3.,0.]]);bh=np.array([.1,-.2])
        after=r*(h@wh.T+bh);before=(r*h)@wh.T+bh
        self.assertGreater(np.max(np.abs(after-before)),1.)
        wi=np.zeros((6,1));weights=np.zeros((6,2));weights[4:]=wh
        bi=np.zeros(6);bi[:2]=np.log(r[0]/(1-r[0]));bias=np.zeros(6);bias[4:]=bh
        _,g=step(np.ones((1,1)),h,wi,weights,bi,bias)
        np.testing.assert_allclose(g['n'],np.tanh(after),atol=1e-14)
        self.assertFalse(np.allclose(g['n'],np.tanh(before)))

    def test_batch_rows_independent(self):
        x,h,*p=worked_parameters();both,_=step(np.repeat(x,2,0),np.repeat(h,2,0),*p)
        single,_=step(x,h,*p);np.testing.assert_allclose(both,np.repeat(single,2,0))

    def test_bad_shapes_rejected(self):
        args=list(worked_parameters());args[2]=np.zeros((2,1))
        with self.assertRaises(ValueError):step(*args)
        args=list(worked_parameters());args[0]=np.zeros((1,1,1))
        with self.assertRaises(ValueError):step(*args)

    def test_nonfinite_and_empty_rejected(self):
        for index in range(6):
            args=[x.copy() for x in worked_parameters()];args[index].flat[0]=np.nan
            with self.assertRaises(ValueError):step(*args)
        args=list(worked_parameters());args[0]=np.empty((0,1))
        with self.assertRaises(ValueError):step(*args)

    def test_gate_domain_rejected(self):
        for z in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError):mix_state(np.zeros(1),np.zeros(1),np.array([z]))

    def test_extreme_sigmoid(self):
        np.testing.assert_equal(sigmoid(np.array([-10000.,10000.])),[0.,1.])

    def test_counts_and_invalid_configuration(self):
        self.assertEqual(parameter_count('lstm',9,16),1728)
        self.assertEqual(parameter_count('gru',9,16),1296)
        for kind,d,h in [('bad',9,16),('gru',0,16),('gru',9,True)]:
            with self.assertRaises(ValueError):parameter_count(kind,d,h)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch parity dependency')
    def test_framework_parity_and_shapes(self):
        from compare_cells import framework_checks
        results=framework_checks()
        self.assertEqual(len(results['parity_cases']),6)
        self.assertEqual(results['shapes']['lstm']['states'],[[1,2,16],[1,2,16]])
        self.assertEqual(results['shapes']['gru']['states'],[[1,2,16]])
        for c in results['parity_cases']:self.assertLess(c['max_abs_error'],1e-12)


if __name__=='__main__':unittest.main()
