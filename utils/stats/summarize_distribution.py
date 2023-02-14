
from typing import List, Dict, Tuple

def frequency(data:List)->Dict:
    """
    Returns a frequency data for the data.
    """
    frequency = {}
    for item in data:
        frequency[item] = frequency.get(item, 0) + 1
    return frequency

def pmf(data:List)->Dict:
    """
    Returns a probability mass function for the data.
    """
    freq = frequency(data)
    n = len(data)
    for x in freq:
        freq[x] /= n
    return freq

# Example
data = [1, 2, 2, 3, 5]
print(pmf(data))