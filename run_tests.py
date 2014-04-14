import unittest
import testing.all_tests

suite = testing.all_tests.create_test_suite()
text_runner = unittest.TextTestRunner().run(suite)
