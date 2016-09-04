import unittest
from GPinv import _version

class test_version(unittest.TestCase):
    def test(self):
        print(_version.__version__)

if __name__ == '__main__':
    unittest.main()
