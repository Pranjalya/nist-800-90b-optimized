import numpy as np

def repetition_count_test(samples, H, alpha=2**-20):
    """
    Performs the Repetition Count Test.

    Args:
        samples (np.ndarray): A NumPy array of noise source samples.
        H (float): The assessed min-entropy per sample.
        alpha (float): The acceptable false-positive probability.

    Returns:
        bool: True if the test passes (no excessive repetitions),
              False if the test fails (excessive repetitions found).
    """

    # 1. Calculate C (Cutoff Value)
    C = int(np.ceil(1 + ((-np.log2(alpha)) / H)))
    # print(C)

    # 2. Repetition Detection (Optimized)
    diffs = np.diff(samples)
    repetition_starts = np.where(diffs == 0)[0]

    if len(repetition_starts) == 0:
        return True  # No repetitions at all

    # Calculate lengths of repetition runs efficiently.
    repetition_lengths = np.diff(repetition_starts)
    # Check for runs >= C.  If any length is NOT 1, it's a run.
    long_runs = repetition_lengths[repetition_lengths != 1]
    
    if len(long_runs)>0:
        for run_length in repetition_lengths:
            if run_length >= C:
                return False
    
    # Check the first and last runs separately.
    if repetition_starts[0] + 1 >= C:
          return False
      
    if len(samples) - repetition_starts[-1] >= C:
          return False

    return True  # Test passed