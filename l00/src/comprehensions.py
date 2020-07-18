"""Tasks for checking knowledge about comprehensions."""
from typing import Dict, List, Tuple


def task_00(numbers: List[int]) -> List[int]:
    """Checks if student can use list comprehension.

    Write a list comprehension on the provided list, so that each new element
    will be multiplied by 2. Only elements that are divisible by 3 should be
    processed by the list comprehension.
    """
    res = [x * 2 for x in numbers if x % 3 == 0]
    return res


def task_01(numbers: List[int]) -> Dict[int, Tuple[int, int]]:
    """Checks if student can use list comprehension.

    Write a dictionary comprehension on the provided list, so that each
    element will result in following mapping: key is the currently processed
    number and the value is a tuple of the currently processed value multiplied
    by 3 and the index of it in the input list.
    """
    res = {x: (x * 3, idx) for idx, x in enumerate(numbers)}
    return res
