"""Test cases for control flow tasks."""
import unittest as ut

from src import control_flow as cf


class TestTask00(ut.TestCase):

    def test_should_return_none_if_x_smaller_than_2(self):
        self.assertEqual(cf.task_00(1), None)

    def test_should_return_int_value_if_x_equal_2(self):
        self.assertEqual(cf.task_00(2), 10)

    def test_should_return_int_value_if_x_greater_than_2(self):
        self.assertEqual(cf.task_00(5), 19)


class TestTask01(ut.TestCase):

    def test_should_return_EMPTY_if_list_is_empty(self):
        self.assertEqual(cf.task_01([]), 'EMPTY')

    def test_should_return_val_if_first_element_greater_than_2(self):
        self.assertEqual(cf.task_01([3]), 9)

    def test_should_return_length_if_first_element_not_greater_than_2(self):
        self.assertEqual(cf.task_01([1, 1, 1]), 3)

