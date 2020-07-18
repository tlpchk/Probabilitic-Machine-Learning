"""Tasks for checking knowledge about functions."""
from typing import List, Callable


def task_00(f: Callable[[int], str], x: int):
    """Checks if student can accept functions as arguments.

    Modify this function signature, so it can accept a function and an integer.
    Call the function with the integer value. Assume the function returns
    a string and add following string to the result: '_END'. Return the whole
    string from this function.
    """
    return f(x) + '_END'


def task_01(nums: List[int]):
    """Checks if student can declare and use nested functions.

    Inside this function declare another function named `absolute`, which
    calculates the absolute value for a given argument. Then use `map` and
    `filter` functions to apply `absolute` on all elements of `nums` and
    remove all values greater or equal to 5.
    """

    def absolute(x):
        return abs(x)

    res = map(absolute, nums)
    res = filter(lambda x: x < 5, res)
    return list(res)
