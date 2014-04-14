import glob
import unittest

def create_test_suite():
    test_file_strings = glob.glob('testing/test_*.py')
    module_strings = ['testing.'+str[8:len(str)-3] for str in test_file_strings]
    suites = [unittest.defaultTestLoader.loadTestsFromName(name) \
              for name in module_strings]
    testSuite = unittest.TestSuite(suites)
    return testSuite
