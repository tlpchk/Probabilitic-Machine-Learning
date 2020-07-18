"""Test cases for classes tasks."""
import abc
import unittest as ut
from unittest.mock import patch, call

from src import klasses as kls


class TestKlasses(ut.TestCase):

    def test_BaseProcessor_should_be_abstract(self):
        self.assertTrue(issubclass(kls.BaseProcessor, abc.ABC))

    def test_process_method_should_be_abstract(self):
        self.assertTrue('process' in kls.BaseProcessor.__abstractmethods__)

    def test_MultiplyingProcessor_inherits_from_BaseProcessor(self):
        self.assertTrue(issubclass(kls.MultiplyingProcessor, kls.BaseProcessor))

    @patch('builtins.print')
    def test_MultiplyingProcessor_prints_str_when_no_items_processed(self, mocked_print):
        mp = kls.MultiplyingProcessor(a=3)
        mp.num_processed()
        self.assertTrue(mocked_print.mock_calls, [call('NO ITEMS PROCESSED')])

    @patch('builtins.print')
    def test_MultiplyingProcessor_no_print_when_items_processed(self, mocked_print):
        mp = kls.MultiplyingProcessor(a=3)
        mp.process(2)
        mp.num_processed()
        self.assertTrue(not mocked_print.called)
