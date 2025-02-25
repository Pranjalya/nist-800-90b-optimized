import unittest
import numpy as np
from suite.iid_tests import (
    conversion_I, 
    conversion_II,
    excursion_test_statistic, 
    number_of_directional_runs, 
    length_of_directional_runs,
    number_of_increases_and_decreases,
    number_of_runs_based_on_median,
    length_of_runs_based_on_median,
    average_collision_test,
    maximum_collision_test_statistic,
    periodicity_test_statistic,
    covariance_test_statistic
)

class TestExcursionTest(unittest.TestCase):

    def test_empty_sequence(self):
        self.assertEqual(excursion_test_statistic([]), 0)

    def test_single_element(self):
        self.assertEqual(excursion_test_statistic([5]), 0)

    def test_example_sequence(self):
        s = [2, 15, 4, 10, 9]
        # Expected average: 8
        # d1 = |2 - 8| = 6
        # d2 = |2 + 15 - 2 * 8| = 1
        # d3 = |2 + 15 + 4 - 3 * 8| = 3
        # d4 = |2 + 15 + 4 + 10 - 4 * 8| = 1
        # d5 = |2 + 15 + 4 + 10 + 9 - 5 * 8| = 0
        # T = max(6, 1, 3, 1, 0) = 6
        self.assertEqual(excursion_test_statistic(s), 6)

    def test_negative_values(self):
        s = [-2, -5, 4, -1, 2]
        # Expected average: -0.4
        # d1 = |-2 - (-0.4)| = 1.6
        # d2 = |-2 - 5 - 2 * (-0.4)| = 6.2
        # d3 = |-2 - 5 + 4 - 3 * (-0.4)| = 1.8
        # d4 = |-2 - 5 + 4 - 1 - 4 * (-0.4)| = 0.6
        # d5 = |-2-5+4-1+2 - 5*(-0.4)| = 0
        # T = max(1.6, 6.2, 1.8, 0.6, 0) = 6.2
        self.assertEqual(excursion_test_statistic(s), 6.2)

    def test_all_same_values(self):
        s = [7, 7, 7, 7, 7]
        self.assertEqual(excursion_test_statistic(s), 0)

    def test_numpy_array_input(self):
        s = np.array([1, 4, 2, 5, 3])
        # Avg = 3
        # d1 = |1 - 3| = 2
        # d2 = |1 + 4 - 2*3| = 1
        # d3 = |1 + 4 + 2 - 3*3| = 2
        # d4 = |1 + 4 + 2 + 5 - 4*3| = 0
        # d5 = |1 + 4 + 2 + 5 + 3 - 5*3| = 0
        # T = max(2, 1, 2, 0, 0) = 2
        self.assertEqual(excursion_test_statistic(s), 2)
    
    def test_float_values(self):
        s = [2.5, 1.2, 3.8, 0.5]
        # Avg = 2.0
        # d1 = |2.5 - 2.0| = 0.5
        # d2 = |2.5 + 1.2 - 2*2.0| = 0.3
        # d3 = |2.5 + 1.2 + 3.8 - 3*2.0| = 1.5
        # d4 = |2.5 + 1.2 + 3.8 + 0.5 - 4*2.0| = 0
        self.assertEqual(excursion_test_statistic(s), 1.5)

class TestConversions(unittest.TestCase):

    def test_conversion_I_empty(self):
        self.assertTrue(np.array_equal(conversion_I(np.array([])), np.array([])))

    def test_conversion_I_full_blocks(self):
        self.assertTrue(np.array_equal(conversion_I(np.array([1,0,0,0,1,1,1,0, 0,1,1,0,0,0,0,1])), np.array([5, 3])))

    def test_conversion_I_partial_block(self):
        self.assertTrue(np.array_equal(conversion_I(np.array([1,0,0,0,1,1,1,0, 0,1,1])), np.array([5, 2])))

    def test_conversion_I_single_block(self):
        self.assertTrue(np.array_equal(conversion_I(np.array([1, 1, 1, 1, 1, 1, 1, 1])), np.array([8])))

    def test_conversion_II_empty(self):
        self.assertTrue(np.array_equal(conversion_II(np.array([])), np.array([])))

    def test_conversion_II_full_blocks(self):
        self.assertTrue(np.array_equal(conversion_II(np.array([1,0,0,0,1,1,1,0, 0,1,1,0,0,0,0,1])), np.array([142, 97])))

    def test_conversion_II_partial_block(self):
        self.assertTrue(np.array_equal(conversion_II(np.array([1,0,0,0,1,1,1,0, 0,1,1])), np.array([142, 48])))

    def test_conversion_II_single_block(self):
        self.assertTrue(np.array_equal(conversion_II(np.array([1, 1, 1, 1, 1, 1, 1, 1])), np.array([255])))

class TestDirectionalRuns(unittest.TestCase):

    def test_number_of_directional_runs_empty(self):
        self.assertEqual(number_of_directional_runs(np.array([])), 0)

    def test_number_of_directional_runs_binary(self):
        self.assertEqual(number_of_directional_runs(np.array([1,0,0,0,1,1,1,0, 0,1,1,0,0,0,0,1]), is_binary=True), 2)
        self.assertEqual(number_of_directional_runs(np.array([1,0,0,0,1,1,1,0, 0,1,1]), is_binary=True), 2)

    def test_number_of_directional_runs_non_binary(self):
        self.assertEqual(number_of_directional_runs(np.array([2, 2, 5, 7, 9, 3, 1, 4, 4])), 3)
        self.assertEqual(number_of_directional_runs(np.array([1, 2, 3, 4, 5])), 1)
        self.assertEqual(number_of_directional_runs(np.array([5, 4, 3, 2, 1])), 1)
        self.assertEqual(number_of_directional_runs(np.array([1, 2, 1, 2, 1])), 4)

    def test_number_of_directional_runs_single_element(self):
        self.assertEqual(number_of_directional_runs(np.array([5])), 0)

    def test_number_of_directional_runs_two_elements(self):
        self.assertEqual(number_of_directional_runs(np.array([5, 2])), 1)
        self.assertEqual(number_of_directional_runs(np.array([2, 5])), 1)

class TestLengthOfDirectionalRuns(unittest.TestCase):

    def test_empty_array(self):
        self.assertEqual(length_of_directional_runs(np.array([])), 0)

    def test_single_element_array(self):
        self.assertEqual(length_of_directional_runs(np.array([5])), 1)

    def test_binary_example(self):
        binary_input = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
        self.assertEqual(length_of_directional_runs(binary_input, binary=True), 3)

    def test_non_binary_example(self):
        non_binary_input = np.array([2, 2, 2, 5, 7, 9, 3, 1, 4, 4])
        self.assertEqual(length_of_directional_runs(non_binary_input), 6)

    def test_all_same_elements(self):
        self.assertEqual(length_of_directional_runs(np.array([5, 5, 5, 5])), 0)

    def test_alternating_elements(self):
        self.assertEqual(length_of_directional_runs(np.array([1, 2, 1, 2, 1])), 1)

    def test_increasing_sequence(self):
        self.assertEqual(length_of_directional_runs(np.array([1, 2, 3, 4, 5])), 4)

    def test_decreasing_sequence(self):
        self.assertEqual(length_of_directional_runs(np.array([5, 4, 3, 2, 1])), 4)

    def test_mixed_sequence_positive_run(self):
        self.assertEqual(length_of_directional_runs(np.array([1, 3, 2, 4, 5, 6, 7, 1])), 4)

    def test_mixed_sequence_negative_run(self):
        self.assertEqual(length_of_directional_runs(np.array([7, 6, 5, 4, 5, 6, 1, 0])), 3)
    
    def test_binary_long_input(self):
        binary_input = np.array([1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(length_of_directional_runs(binary_input, binary=True), 6)

    def test_non_binary_with_zeros(self):
        self.assertEqual(length_of_directional_runs(np.array([0, 0, 1, 2, 0, 0])), 2)

class TestIncreasesDecreases(unittest.TestCase):

    def test_empty_array(self):
        self.assertEqual(number_of_increases_and_decreases(np.array([])), 0)
        self.assertEqual(number_of_increases_and_decreases(np.array([]), binary=True), 0)

    def test_single_element_array(self):
        self.assertEqual(number_of_increases_and_decreases(np.array([5])), 0)
        self.assertEqual(number_of_increases_and_decreases(np.array([1]), binary=True), 0)

    def test_example_case(self):
        arr = np.array([2, 2, 2, 5, 7, 7, 9, 3, 1, 4, 4])
        self.assertEqual(number_of_increases_and_decreases(arr), 8)

    def test_binary_example(self):
        binary_arr = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1])  # Example from prompt
        self.assertEqual(number_of_increases_and_decreases(binary_arr, binary=True), 2)

    def test_all_same_elements(self):
        self.assertEqual(number_of_increases_and_decreases(np.array([5, 5, 5, 5])), 0)
        self.assertEqual(number_of_increases_and_decreases(np.array([1, 1, 1, 1, 1, 1, 1, 1]), binary=True), 0)

    def test_increasing_sequence(self):
        self.assertEqual(number_of_increases_and_decreases(np.array([1, 2, 3, 4, 5])), 4)

    def test_decreasing_sequence(self):
        self.assertEqual(number_of_increases_and_decreases(np.array([5, 4, 3, 2, 1])), 4)

    def test_mixed_sequence(self):
        self.assertEqual(number_of_increases_and_decreases(np.array([1, 3, 2, 4, 5, 3, 2, 1])), 4)

    def test_binary_long_sequence(self):
        binary_arr = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1])
        self.assertEqual(number_of_increases_and_decreases(binary_arr, binary=True), 1)

    def test_binary_all_zeros(self):
        binary_arr = np.zeros(20, dtype=int)
        self.assertEqual(number_of_increases_and_decreases(binary_arr, binary=True), 0)

    def test_binary_all_ones(self):
        binary_arr = np.ones(20, dtype=int)
        self.assertEqual(number_of_increases_and_decreases(binary_arr, binary=True), 0)
    
    def test_non_binary_with_negative_numbers(self):
        arr = np.array([-2, -1, 0, 1, 0, -1, -2])
        self.assertEqual(number_of_increases_and_decreases(arr), 4)

    def test_non_binary_with_large_numbers(self):
        arr = np.array([1000, 2000, 1500, 2500, 3000])
        self.assertEqual(number_of_increases_and_decreases(arr), 3)

class TestRunsNumMedian(unittest.TestCase):

    def test_non_binary_example(self):
        arr = np.array([5, 15, 12, 1, 13, 9, 4])
        expected_runs = 5
        actual_runs = number_of_runs_based_on_median(arr)
        self.assertEqual(actual_runs, expected_runs)

    def test_binary_example(self):
        arr = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0])
        expected_runs = 7
        actual_runs = number_of_runs_based_on_median(arr, binary=True)
        self.assertEqual(actual_runs, expected_runs)

    def test_empty_array(self):
        arr = np.array([])
        expected_runs = 1  # An empty array should have one run (no changes)
        actual_runs = number_of_runs_based_on_median(arr)
        self.assertEqual(actual_runs, expected_runs)

        expected_runs = 1
        actual_runs = number_of_runs_based_on_median(arr, binary=True)
        self.assertEqual(actual_runs, expected_runs)


    def test_all_same_non_binary(self):
        arr = np.array([5, 5, 5, 5, 5])
        expected_runs = 1  # All same elements, one run
        actual_runs = number_of_runs_based_on_median(arr)
        self.assertEqual(actual_runs, expected_runs)

    def test_all_same_binary(self):
        arr = np.array([1, 1, 1, 1, 1])
        expected_runs = 1
        actual_runs = number_of_runs_based_on_median(arr, binary=True)
        self.assertEqual(actual_runs, expected_runs)

    def test_alternating_non_binary(self):
        arr = np.array([1, 2, 1, 2, 1, 2])
        expected_runs = 6
        actual_runs = number_of_runs_based_on_median(arr)
        self.assertEqual(actual_runs, expected_runs)

    def test_alternating_binary(self):
        arr = np.array([0, 1, 0, 1, 0, 1])
        expected_runs = 6
        actual_runs = number_of_runs_based_on_median(arr, binary=True)
        self.assertEqual(actual_runs, expected_runs)

    def test_non_binary_with_duplicates_at_median(self):
        arr = np.array([5, 10, 7, 7, 12, 7, 3])  # Median is 7
        expected_runs = 5  # [-1, 1, -1, -1, 1, -1, -1] -> [1, -1, 1, -1] -> 5 runs
        actual_runs = number_of_runs_based_on_median(arr)
        self.assertEqual(actual_runs, expected_runs)
    
    def test_binary_long(self):
        arr = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        expected_runs = 8
        actual_runs = number_of_runs_based_on_median(arr, binary=True)
        self.assertEqual(actual_runs, expected_runs)

class TestRunsLengthMedian(unittest.TestCase):

    def test_empty_array(self):
        self.assertEqual(length_of_runs_based_on_median(np.array([])), 0)

    def test_single_element_array(self):
        self.assertEqual(length_of_runs_based_on_median(np.array([5])), 1)

    def test_binary_array(self):
        arr = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        self.assertEqual(length_of_runs_based_on_median(arr, is_binary=True), 2)

    def test_non_binary_array(self):
        arr = np.array([5, 15, 12, 1, 13, 9, 4])
        self.assertEqual(length_of_runs_based_on_median(arr), 2)

    def test_all_same_values(self):
        arr = np.array([7, 7, 7, 7, 7])
        self.assertEqual(length_of_runs_based_on_median(arr), 5)

    def test_alternating_values(self):
        arr = np.array([1, 2, 1, 2, 1, 2])
        self.assertEqual(length_of_runs_based_on_median(arr), 1)
    
    def test_binary_array_long_run(self):
        arr = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(length_of_runs_based_on_median(arr, is_binary=True), 7)

    def test_non_binary_array_long_run(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3])
        self.assertEqual(length_of_runs_based_on_median(arr), 7)

    def test_non_binary_array_with_negative_numbers(self):
        arr = np.array([-5, -15, -12, 1, 13, 9, 4])
        self.assertEqual(length_of_runs_based_on_median(arr), 3)

    def test_binary_array_with_only_zeros(self):
        arr = np.array([0, 0, 0, 0, 0])
        self.assertEqual(length_of_runs_based_on_median(arr, is_binary=True), 5)

    def test_binary_array_with_only_ones(self):
        arr = np.array([1, 1, 1, 1, 1])
        self.assertEqual(length_of_runs_based_on_median(arr, is_binary=True), 5)

class TestAverageCollision(unittest.TestCase):

    def test_no_collision(self):
        arr = np.array([1, 2, 3, 4, 5])
        self.assertEqual(average_collision_test(arr), np.inf)

    def test_example_case(self):
        arr = np.array([2, 1, 1, 2, 0, 1, 0, 1, 1, 2])
        self.assertEqual(average_collision_test(arr), 3.0)

    def test_all_same(self):
        arr = np.array([5, 5, 5, 5, 5])
        self.assertEqual(average_collision_test(arr), 2.0)  # First collision at j=2

    def test_binary_input(self):
        arr = np.array([1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1])  # Example from the document
        # After conversion_II: [143, 170, 51] -> expecting collision at j=2
        self.assertEqual(average_collision_test(arr, is_binary=True), np.inf)
    
    def test_binary_input_collision(self):
        arr = np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
        # After conversion_II: [143, 143, 51]
        self.assertEqual(average_collision_test(arr, is_binary=True), 2.0)

    def test_empty_array(self):
        arr = np.array([])
        self.assertEqual(average_collision_test(arr), np.inf)
        
    def test_one_element_array(self):
        arr = np.array([5])
        self.assertEqual(average_collision_test(arr), np.inf)

    def test_binary_empty_array(self):
        arr = np.array([])
        self.assertEqual(average_collision_test(arr, is_binary=True), np.inf)

    def test_long_array_with_late_collision(self):
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 11])
        self.assertEqual(average_collision_test(arr), 11)

    def test_binary_input_long(self):
        # Create a long binary array where collision will happen after conversion
        arr = np.tile(np.array([1,0,0,0,1,1,1,1]), 20)  # Repeat [1,0,0,0,1,1,1,1] 20 times
        arr = np.concatenate((arr, np.array([1,0,0,0,1,1,1,1]))) # Add one more for collision
        self.assertEqual(average_collision_test(arr, is_binary=True), 2.0)


class TestMaximumCollision(unittest.TestCase):

    def test_empty_array(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([])), 0)

    def test_no_collision(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([1, 2, 3, 4, 5])), 0)

    def test_single_collision(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([1, 2, 3, 1, 5])), 3)

    def test_multiple_collisions(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([2, 1, 1, 2, 0, 1, 0, 1, 1, 2])), 4)

    def test_all_same(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([5, 5, 5, 5, 5])), 2)

    def test_binary_no_collision(self):
        binary_input = np.array([0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])  # No collision after conversion
        self.assertEqual(maximum_collision_test_statistic(binary_input, is_binary=True), 0)

    def test_binary_with_collision(self):
        binary_input = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1])  # 129, 129
        self.assertEqual(maximum_collision_test_statistic(binary_input, is_binary=True), 2)
    
    def test_binary_long(self):
        binary_input = np.array([1, 0, 0, 0, 0, 0, 0, 1,
                                 1, 0, 0, 0, 0, 0, 0, 1,
                                 0, 1, 0, 0, 0, 0, 0, 1,
                                 0, 1, 0, 0, 0, 0, 0, 0,])  # 129, 129, 65, 64
        self.assertEqual(maximum_collision_test_statistic(binary_input, is_binary=True), 2)

    def test_non_binary_large_numbers(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([1000, 2000, 1000, 3000])), 2)

    def test_non_binary_mixed(self):
        self.assertEqual(maximum_collision_test_statistic(np.array([1, 2, 1, 4, 5, 4, 7, 8, 9, 10])), 2)



def periodicity_test(arr, p_values=None, is_binary=False):
    """
    Performs the periodicity test for multiple lag values.

    Args:
        arr: The input array.
        p_values: A list of lag values to test.  Defaults to [1, 2, 8, 16, 32].
        is_binary: Whether the input is binary.

    Returns:
        A dictionary mapping lag values to their corresponding test statistics.
    """
    if p_values is None:
        p_values = [1, 2, 8, 16, 32]

    results = {}
    for p in p_values:
        results[p] = periodicity_test_statistic(arr, p, is_binary)
    return results



class TestPeriodicity(unittest.TestCase):

    def test_empty_array(self):
        with self.assertRaises(ValueError):
            periodicity_test_statistic(np.array([]), 1, is_binary=False)
        with self.assertRaises(ValueError):
            periodicity_test_statistic(np.array([]), 1, is_binary=True)


    def test_invalid_p(self):
        with self.assertRaises(ValueError):
            periodicity_test_statistic(np.array([1, 2, 3]), 0, is_binary=False)  # p = 0
        with self.assertRaises(ValueError):
            periodicity_test_statistic(np.array([1, 2, 3]), 3, is_binary=False)  # p = L
        with self.assertRaises(ValueError):
            periodicity_test_statistic(np.array([1, 2, 3]), 4, is_binary=False)  # p > L

    def test_binary_example(self):
        arr = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        # conversion_I result: [4, 6, 2]
        self.assertEqual(periodicity_test_statistic(arr, 1, is_binary=True), 0)
        self.assertEqual(periodicity_test_statistic(arr, 2, is_binary=True), 1)

    def test_non_binary_example(self):
        arr = np.array([2, 1, 2, 1, 0, 1, 0, 1, 1, 2])
        self.assertEqual(periodicity_test_statistic(arr, 2, is_binary=False), 5)
        self.assertEqual(periodicity_test_statistic(arr, 3, is_binary=False), 2)

    def test_binary_all_same(self):
        arr = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(periodicity_test_statistic(arr, 1, is_binary=True), 0) # [8] after conversion
        self.assertEqual(periodicity_test_statistic(arr, 2, is_binary=True), 0)
        self.assertEqual(periodicity_test_statistic(arr, 3, is_binary=True), 0)

    def test_non_binary_all_same(self):
        arr = np.array([5, 5, 5, 5, 5])
        self.assertEqual(periodicity_test_statistic(arr, 1, is_binary=False), 4)
        self.assertEqual(periodicity_test_statistic(arr, 2, is_binary=False), 3)
        self.assertEqual(periodicity_test_statistic(arr, 4, is_binary=False), 1)

    def test_binary_alternating(self):
        arr = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # [1,1,1,1]
        self.assertEqual(periodicity_test_statistic(arr, 1, is_binary=True), 3)
        self.assertEqual(periodicity_test_statistic(arr, 2, is_binary=True), 2)

    def test_non_binary_alternating(self):
        arr = np.array([1, 2, 1, 2, 1, 2])
        self.assertEqual(periodicity_test_statistic(arr, 1, is_binary=False), 0)
        self.assertEqual(periodicity_test_statistic(arr, 2, is_binary=False), 4)
        self.assertEqual(periodicity_test_statistic(arr, 3, is_binary=False), 0)

    def test_periodicity_test_default_p(self):
        arr = np.array([2, 1, 2, 1, 0, 1, 0, 1, 1, 2])
        expected_results = {1: 2, 2: 5, 8: 0, 16: 0, 32: 0}  # Manually calculated
        actual_results = periodicity_test(arr)
        self.assertEqual(actual_results, expected_results)

    def test_periodicity_test_custom_p(self):
        arr = np.array([2, 1, 2, 1, 0, 1, 0, 1, 1, 2])
        p_values = [1, 3, 5]
        expected_results = {1: 2, 3: 2, 5: 1}  # Manually calculated
        actual_results = periodicity_test(arr, p_values=p_values)
        self.assertEqual(actual_results, expected_results)

    def test_periodicity_test_binary(self):
        arr = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]) # [4, 6, 2]
        expected_results = {1: 0, 2: 1, 8: 0, 16: 0, 32: 0}
        actual_results = periodicity_test(arr, is_binary=True)
        self.assertEqual(actual_results, expected_results)

def covariance_test_statistic_normal(arr, p, is_binary=False):
    """
    Calculates the covariance test statistic.

    Args:
        arr (np.ndarray): Input array.
        p (int): Lag value (p < L, where L is the length of the array).
        is_binary (bool): True if the input array is binary, False otherwise.

    Returns:
        float: The covariance test statistic.
    """

    if is_binary:
        arr = conversion_I(arr)

    L = len(arr)
    if p >= L:
        raise ValueError("p must be less than the length of the array")

    T = 0
    for i in range(L - p):
        T += arr[i] * arr[i + p]
    return T



class TestCovarianceTest(unittest.TestCase):

    def test_non_binary_data(self):
        arr = np.array([5, 2, 6, 10, 12, 3, 1])
        p = 2
        expected_result = 164
        result = covariance_test_statistic(arr, p, is_binary=False)
        self.assertEqual(result, expected_result)

        arr = np.array([5, 2, 6, 10, 12, 3, 1])
        p = 2
        expected_result = 164
        result = covariance_test_statistic_normal(arr, p, is_binary=False)
        self.assertEqual(result, expected_result)


    def test_binary_data(self):
        arr = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        p = 3
        expected_result = 14 # (4 * 6) + (6 * 2)
        result = covariance_test_statistic(arr, p, is_binary=True)
        self.assertEqual(result, 14)

        arr = np.array([1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
        p = 3
        expected_result = 14
        result = covariance_test_statistic_normal(arr, p, is_binary=True)
        self.assertEqual(result, 14)

    def test_binary_data_short(self):
        arr = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        p = 1
        expected_result = 0
        result = covariance_test_statistic(arr, p, is_binary=True)
        self.assertEqual(result, expected_result)

        arr = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        p = 1
        expected_result = 0
        result = covariance_test_statistic_normal(arr, p, is_binary=True)
        self.assertEqual(result, expected_result)

    def test_p_equal_L(self):
        arr = np.array([1, 2, 3, 4, 5])
        p = 5  # p = L
        with self.assertRaises(ValueError):
            covariance_test_statistic(arr, p, is_binary=False)

    def test_p_greater_than_L(self):
        arr = np.array([1, 2, 3, 4, 5])
        p = 6  # p > L
        with self.assertRaises(ValueError):
            covariance_test_statistic(arr, p, is_binary=False)

    def test_empty_array(self):
        arr = np.array([])
        p = 2
        with self.assertRaises(ValueError):
            covariance_test_statistic(arr, p, is_binary=False)

    def test_p_zero(self):
        arr = np.array([1, 2, 3, 4, 5])
        p = 0  # p = 0
        expected_result = 55 # 1*1 + 2*2 + 3*3 + 4*4 + 5*5
        result = covariance_test_statistic(arr, p, is_binary=False)
        self.assertEqual(result, expected_result)

        arr = np.array([1, 2, 3, 4, 5])
        p = 0  # p = 0
        expected_result = 55
        result = covariance_test_statistic_normal(arr, p, is_binary=False)
        self.assertEqual(result, expected_result)

    def test_one_element_array(self):
        arr = np.array([5])
        p = 0
        expected_result = 25
        result = covariance_test_statistic(arr,p, is_binary= False)
        self.assertEqual(result, expected_result)

        arr = np.array([5])
        p = 0
        expected_result = 25
        result = covariance_test_statistic_normal(arr,p, is_binary= False)
        self.assertEqual(result, expected_result)

    def test_one_element_array_binary(self):
        arr = np.array([1])
        p=0
        expected_result = 1
        result = covariance_test_statistic_normal(arr, p, is_binary=True)
        self.assertEqual(result, expected_result)

        arr = np.array([1])
        p=0
        expected_result = 1
        result = covariance_test_statistic(arr, p, is_binary=True)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()