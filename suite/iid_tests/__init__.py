import numpy as np
from suite.iid_tests.utils import conversion_I, conversion_II


def excursion_test_statistic(s):
    """
    Calculates the excursion test statistic (T) for a given sequence (s).

    The excursion test statistic measures how far the running sum of sample
    values deviates from its average value at each point in the dataset.

    Args:
        s (array-like):  The input sequence of sample values.  Can be a list
                        or a NumPy array.

    Returns:
        float: The excursion test statistic (T), which is the maximum absolute
               deviation.
    """

    L = len(s)
    x_bar = np.mean(s)
    d = np.zeros(L)

    # More efficient cumsum calculation, and vectorized d calculation.
    cumulative_sums = np.cumsum(s)
    d = np.abs(cumulative_sums - np.arange(1, L + 1) * x_bar)
    T = np.max(d)
    return T




def number_of_directional_runs(arr, is_binary=False):
    """
    Calculates the number of directional runs in a sequence.

    Args:
        arr (np.ndarray): Input sequence.
        is_binary (bool): True if the input is binary, False otherwise.

    Returns:
        int: The number of directional runs.
    """
    if is_binary:
        arr = conversion_I(arr)

    s_prime = np.where(arr[:-1] <= arr[1:], 1, -1)
    
    runs = 0
    if len(s_prime) > 0:
        runs = 1
        for i in range(len(s_prime) - 1):
            if s_prime[i] != s_prime[i+1]:
                runs += 1
    return runs


def length_of_directional_runs(arr, binary=False):
    """
    Calculates the length of the longest directional run in a sequence.

    Args:
        arr (np.ndarray): Input array (numeric).
        binary (bool): True if the input is binary, False otherwise.

    Returns:
        int: The length of the longest directional run.
    """
    if binary:
        arr = conversion_I(arr)

    if len(arr) <= 1:  # Handle edge cases of 0 or 1 element
        return 0 if len(arr) == 0 else 1 # If there is only 1 element, there is no run. Only a single element.

    s_prime = np.where(np.diff(arr) > 0, 1, -1)
    
    max_run_length = 0
    current_run_length = 0
    
    # Iterate over the s_prime array to calculate run lengths
    for x in s_prime:
        if x == 1:
            if current_run_length >= 0:  # Continue positive run
                current_run_length += 1
            else:  # Start new positive run
                current_run_length = 1
        else: # x == -1
            if current_run_length <= 0:  # Continue negative run
                current_run_length -= 1
            else:  # Start new negative run
                current_run_length = -1
        max_run_length = max(max_run_length, abs(current_run_length))

    return max_run_length



def number_of_increases_and_decreases(arr, binary=False):
    """
    Calculates the maximum number of increases or decreases between consecutive sample values.

    Args:
        arr (np.ndarray): Input array.
        binary (bool): True if the input array is binary, False otherwise.

    Returns:
        int: The maximum number of increases or decreases.
    """
    if binary:
        arr = conversion_I(arr)

    # Efficiently calculate S' using NumPy's diff and sign functions
    s_prime = np.sign(np.diff(arr))

    # Count the number of -1s and 1s, handling edge cases of empty arrays
    num_neg_ones = np.count_nonzero(s_prime == -1) if s_prime.size > 0 else 0
    num_pos_ones = np.count_nonzero(s_prime == 1) if s_prime.size > 0 else 0

    return max(num_neg_ones, num_pos_ones)



def number_of_runs_based_on_median(arr, binary=False):
    """
    Calculates the number of runs based on the median of the input array.

    Args:
        arr (np.ndarray): Input array.
        binary (bool): True if the input array is binary, False otherwise.

    Returns:
        int: The number of runs.
    """
    if binary:
        median = 0.5
        s_prime = np.where(arr < median, -1, 1)
    else:
        median = np.median(arr)
        s_prime = np.where(arr < median, -1, 1)

    # Count runs.  np.diff finds the difference between consecutive elements.
    #  A change from -1 to 1 or 1 to -1 indicates a run change (difference of 2 or -2).
    runs = 1 + np.count_nonzero(np.diff(s_prime))
    return runs



def length_of_runs_based_on_median(arr: np.ndarray, is_binary: bool = False) -> int:
    """
    Calculates the length of the longest run of values above/below the median.

    Args:
        arr: The input NumPy array.
        is_binary: True if the input array is binary, False otherwise.

    Returns:
        The length of the longest run.
    """

    if is_binary:
        median = 0.5
        s_prime = np.where(arr > median, 1, -1)  # Use -1 and 1, not 0 and 1
    else:
        median = np.median(arr)
        s_prime = np.where(arr >= median, 1, -1)

    max_run_length = 0
    current_run_length = 0
    if s_prime.size > 0 :
        current_value = s_prime[0]  # Initialize with the first value

        for value in s_prime:
            if value == current_value:
                current_run_length += 1
            else:
                max_run_length = max(max_run_length, current_run_length)
                current_run_length = 1
                current_value = value

        max_run_length = max(max_run_length, current_run_length)  # Check the last run

    return max_run_length


def average_collision_test(arr, is_binary=False):
    """
    Calculates the average collision test statistic.

    Args:
        arr (np.ndarray): Input array.
        is_binary (bool): True if the input is binary, False otherwise.

    Returns:
        float: The average collision test statistic.  Returns np.inf if no collisions occur.
    """
    if is_binary:
        arr = conversion_II(arr)

    c = []
    i = 0
    while i < len(arr):
        for j in range(1, len(arr) - i + 1):
            sub_array = arr[i:i + j]
            if len(sub_array) != len(np.unique(sub_array)):  # Check for duplicates
                c.append(j)
                i += j
                break
        else:  # No collision found for this i
            break  # Exit the outer loop as well

    return np.mean(c) if c else np.inf



def maximum_collision_test_statistic(arr, is_binary=False):
    """
    Calculates the maximum collision test statistic.

    Args:
        arr (np.ndarray): Input array.
        is_binary (bool): True if the input is binary, False otherwise.

    Returns:
        int: The maximum collision test statistic.
    """
    if is_binary:
        arr = conversion_II(arr)

    c = []
    i = 0
    while i < len(arr):
        j = 1
        found_duplicate = False
        while i + j <= len(arr):
            sub_array = arr[i:i + j]
            # Check for duplicates in the sub-array using sets for efficiency
            if len(sub_array) != len(set(sub_array)):
                found_duplicate = True
                break  # Exit inner loop as soon as a duplicate is found
            j += 1
        
        if found_duplicate:
             c.append(j)
             i += j
        else:
            break # No more duplicates found, exit the main loop.

    return max(c) if c else 0  # Handle the case where no collisions are found



def periodicity_test_statistic(arr, p, is_binary=False):
    """
    Calculates the periodicity test statistic T for a given array and lag parameter p.

    Args:
        arr (np.ndarray): Input array (binary or non-binary).
        p (int): Lag parameter (p < L, where L is the length of the array).
        is_binary (bool): True if the input array is binary, False otherwise.

    Returns:
        int: The periodicity test statistic T.
    """

    if is_binary:
        arr = conversion_I(arr)

    L = len(arr)
    if not (0 < p < L):
        raise ValueError("p must be between 0 and L (length of array)")

    T = 0
    for i in range(L - p):
        if arr[i] == arr[i + p]:
            T += 1
    return T



def covariance_test_statistic(arr, p, is_binary=False):
    """
    Calculates the covariance test statistic (optimized version).

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

    return np.sum(arr[:-p] * arr[p:])

