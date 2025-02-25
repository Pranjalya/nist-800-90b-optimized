import numpy as np


def convert_to_numpy_array(s):
    return np.asarray(s)


def conversion_I(arr):
    """
    Partitions the input array into eight-bit non-overlapping blocks and counts the number of ones in each block.
    Zeroes are appended when the last block has less than eight bits.

    Args:
        arr (np.ndarray): Input array (binary).

    Returns:
        np.ndarray: Array containing the counts of ones in each block.
    """
    padding_needed = 8 - len(arr) % 8
    if padding_needed != 8:  # not full last block
        padded_arr = np.pad(arr, (0, padding_needed), 'constant')
    else:
        padded_arr = arr
    
    reshaped_arr = padded_arr.reshape(-1, 8)
    return np.sum(reshaped_arr, axis=1)


def conversion_II(arr):
    """
    Partitions the sequences into eight-bit non-overlapping blocks and calculates the integer value of each block.
    Zeroes are appended when the last block has less than eight bits.

    Args:
        arr (np.ndarray): Input array (binary).

    Returns:
        np.ndarray: Array containing the integer values of each block.
    """
    padding_needed = 8 - len(arr) % 8
    if padding_needed != 8:
        padded_arr = np.pad(arr, (0, padding_needed), 'constant')
    else:
        padded_arr = arr
    reshaped_arr = padded_arr.reshape(-1, 8)
    powers_of_two = 2 ** np.arange(8)[::-1]  # Reverse for correct bit order [128, 64, 32, 16, 8, 4, 2, 1]
    return np.sum(reshaped_arr * powers_of_two, axis=1)