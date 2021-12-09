# -*- coding: utf-8 -*-
"""
Test module for data input

@author: annam
"""

import unittest

class TestStringMethods(unittest.TestCase):
    # test function
    def test_negative(self):
        testValue = False
        # error message in case if test case got failed
        message = "Test when input is not true."
        # assertTrue() to check true of test value
        self.assertTrue( testValue, message)
 
if __name__ == '__main__':
    unittest.main()
    
class TestStringMethods(unittest.TestCase):
    # test function 
    def test_positive(self):
        testValue = False
        # error message in case if test case got failed
        message = "Test when input is not false."
        # assertFalse() to check test value as false
        self.assertFalse( testValue, message)
  
if __name__ == '__main__':
    unittest.main()

