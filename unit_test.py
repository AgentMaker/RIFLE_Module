import unittest
from dev import main


class MyTestCase(unittest.TestCase):
    def test_none_init(self):
        main()

    def test_init(self):
        main(True)


if __name__ == '__main__':
    unittest.main()
