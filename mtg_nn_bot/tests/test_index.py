# -*- coding: utf-8 -*-
import unittest
from .. import INDEX


class ModelTests(unittest.TestCase):
    def test_generate_boosters(self):
        for expansion in INDEX.get_draftable_expansions():
            INDEX.get_expansion(expansion).generate_booster()


if __name__ == "__main__":
    unittest.main()