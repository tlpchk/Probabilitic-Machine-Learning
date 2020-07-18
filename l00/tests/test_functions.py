"""Test cases for functions tasks."""
import unittest as ut

from src import functions as fns


class TestTask00(ut.TestCase):

    def test_should_return_only_END_if_called_with_zero(self):
        self.assertEqual(fns.task_00(lambda x: 'THE' * x, 0), '_END')

    def test_should_return_complete_str_if_called_with_one(self):
        self.assertEqual(fns.task_00(lambda x: 'THE' * x, 1), 'THE_END')

    def test_should_return_repeated_str_if_called_with_two(self):
        self.assertEqual(fns.task_00(lambda x: 'THE' * x, 2), 'THETHE_END')


class TestTask01(ut.TestCase):

    def test_should_return_empty_list_if_input_empty(self):
        self.assertEqual(fns.task_01([]), [])

    def test_should_return_only_nonnegative_values(self):
        self.assertTrue(all(n >= 0 for n in fns.task_01([0, -1, 2, -3])))

    def test_should_return_only_values_less_than_5(self):
        self.assertTrue(all(n < 5 for n in fns.task_01([-1, -2, 50, 5])))
