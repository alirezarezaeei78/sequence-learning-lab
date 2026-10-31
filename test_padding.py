"""Focused checks for actual padding/masking failure modes."""
import unittest
from unittest.mock import patch
import importlib.metadata
import torch
from padding_demo import example,worked_lstm,lengths_for,packed


class PaddingChecks(unittest.TestCase):
    def test_independent_scalar_and_standalone(self):
        d=example();self.assertLess(d['independent_scalar_max_abs_h_error'],1e-14)
        self.assertEqual(d['inputs']['valid_mask'],[[True,True,False,False],[True]*4])
        self.assertTrue(d['packed_padded_input_gradients_zero'])
        for i in range(2):self.assertAlmostEqual(d['packed_final_h'][i],d['standalone'][i]['h'],places=14)
    def test_padding_is_not_a_stop_instruction(self):
        d=example();self.assertGreater(abs(d['ordinary_final_h'][0]-d['packed_final_h'][0]),.01)
        self.assertNotEqual(d['ordinary_final_h'][0],d['padding_changed_to9']['ordinary_final_h'][0])
        self.assertEqual(d['loss_mask']['valid_terms'],6)
        self.assertNotEqual(d['loss_mask']['masked'],d['loss_mask']['naive_all_padded'])
    def test_lengths_reject_bad_values(self):
        x=torch.zeros((2,4,1),dtype=torch.float64)
        for lengths in [[0,4],[-1,4],[5,4],[2.,4],[True,4],[2],None]:
            with self.assertRaises(ValueError):lengths_for(x,lengths)
        with self.assertRaises(ValueError):lengths_for(torch.zeros(2,4),[2,4])
    def test_unsorted_packing_restores_original_order(self):
        m=worked_lstm();x=torch.tensor([[[1.],[2.],[0.],[0.]],[[0.],[1.],[-1.],[2.]]],dtype=torch.float64)
        h=packed(m,x,[2,4])[1][0]
        for i,n in enumerate([2,4]):torch.testing.assert_close(h[:,i],m(x[i:i+1,:n])[1][0][:,0],atol=1e-14,rtol=0)
    def test_extra_padding_and_all_equal_lengths(self):
        m=worked_lstm();x=torch.ones((2,2,1),dtype=torch.float64);extra=torch.cat([x,torch.full((2,5,1),9.,dtype=torch.float64)],1)
        h=packed(m,x,[2,2])[1][0]
        torch.testing.assert_close(h,packed(m,extra,[2,2])[1][0],atol=1e-14,rtol=0)
        torch.testing.assert_close(h,m(x)[1][0],atol=1e-14,rtol=0)
    def test_truncation_and_wrong_lengths_remove_information(self):
        d=example();self.assertNotAlmostEqual(d['packed_final_h'][1],d['truncation']['B_first_two_final_h'],places=8)
        self.assertNotAlmostEqual(d['packed_final_h'][0],d['truncation']['A_incorrect_length1_final_h'],places=8)
    def test_numeric_runner_without_optional_plotting_dependency(self):
        real_version=importlib.metadata.version
        def minimal_environment(name):
            if name=='matplotlib':raise importlib.metadata.PackageNotFoundError(name)
            return real_version(name)
        with patch('importlib.metadata.version',side_effect=minimal_environment):
            result=example()
        self.assertIsNone(result['environment']['dependencies']['matplotlib'])
        self.assertAlmostEqual(result['packed_final_h'][0],.5553963082983773,places=14)


if __name__=='__main__':unittest.main()
