"""Test cases for comprehensions tasks."""
import unittest as ut

from src import comprehensions as com


class TestTask00(ut.TestCase):

    def test_should_process_elements(self):
        self.assertEqual(com.task_00([1, 2, 3, 4, 5, 6]), [6, 12])


class TestTask01(ut.TestCase):

    def test_should_process_elements(self):
        self.assertEqual(
            com.task_01([6, 7, 8]),
            {6: (18, 0), 7: (21, 1), 8: (24, 2)}
        )
