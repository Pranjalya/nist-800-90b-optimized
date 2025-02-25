import unittest
import numpy as np
from suite.continous_health_test import repetition_count_test

class TestRepetitionCountTest(unittest.TestCase):

    def test_no_repetitions(self):
        samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.assertTrue(repetition_count_test(samples, H=2.0))

    def test_short_repetitions(self):
        samples = np.array([1, 1, 2, 3, 4, 4, 5, 6, 7, 8])  # Repetitions of length 2
        self.assertTrue(repetition_count_test(samples, H=2.0))  # Should pass (C = 11)

    def test_long_repetition_failure(self):
        samples = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3])  # Repetition of length 11
        self.assertFalse(repetition_count_test(samples, H=2.0))  # Should fail (C = 11)

    def test_long_repetition_at_end_failure(self):
        samples = np.array([2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Repetition of length 11 at the end
        self.assertFalse(repetition_count_test(samples, H=2.0))  # Should fail

    def test_edge_case_c_equals_length(self):
        samples = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])  # Repetition of length 11
        self.assertFalse(repetition_count_test(samples, H=2.0))  # Should fail (C = 11)

    def test_different_h(self):
        samples = np.array([1, 1, 1, 1, 1, 2, 3, 4, 5, 6])  # Repetition of length 5
        self.assertTrue(repetition_count_test(samples, H=4.0))  # Should pass (C = 6)
        self.assertFalse(repetition_count_test(samples, H=1.0)) # Should fail (C=21)

    def test_different_alpha(self):
        samples = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 3])  # Repetition of length 8
        self.assertTrue(repetition_count_test(samples, H=2.0, alpha=2**-10))  # Should pass (C = 6)
        self.assertFalse(repetition_count_test(samples, H=2.0, alpha=2**-30)) # Should fail (C = 16)
    
    def test_empty_array(self):
        samples = np.array([])
        self.assertTrue(repetition_count_test(samples, H=2.0))

    def test_single_element_array(self):
        samples = np.array([1])
        self.assertTrue(repetition_count_test(samples, H=2.0))
        
    def test_two_same_elements(self):
        samples = np.array([7, 7])
        self.assertTrue(repetition_count_test(samples, H=2.0)) # C = 11, so this passes
        self.assertFalse(repetition_count_test(samples, H=0.1)) # C = 201, so this fails

if __name__ == '__main__':
    unittest.main()