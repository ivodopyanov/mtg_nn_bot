# -*- coding: utf-8 -*-
import os
import unittest

from mtg_nn_bot.service.draft_controller import DraftController
from .. import DIR

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = DraftController(os.path.join(DIR, "models","IXA_IXA_IXA"))


    def test1(self):
        draft1 = self.processor.start_draft([False,True,True,True,True,True,True,True], 1)
        draft2 = self.processor.start_draft([True,True,True,True,True,True,True,True], 2)
        self.processor.predict([draft1, draft2])
        self.log_draft(draft1, os.path.join(DIR, "processor_tests", "draft1.txt"))
        self.log_draft(draft2, os.path.join(DIR, "processor_tests", "draft2.txt"))

    def log_draft(self, draft, path):
        f = open(path, "wt")
        for pack_num in range(self.processor.model.settings['round_num']):
            for pick_num, packs in enumerate(draft.iterate_over_packs_in_round(pack_num)):
                card_names = []
                for player_num, player in enumerate(draft.players):
                    chosen_card = draft.picks[pack_num, player_num, pick_num] - 1
                    if chosen_card == -1:
                        card_names.append("empty")
                    else:
                        if packs[player_num, chosen_card] == 0:
                            raise Exception("Pack {} pick {} player {}".format(pack_num, pick_num, player_num))
                        card_names.append(self.processor.index[packs[player_num, chosen_card]-1])
                f.write("{}\n".format("||".join(card_names)))
        f.close()



if __name__ == "__main__":
    unittest.main()