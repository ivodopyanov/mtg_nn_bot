# -*- coding: utf-8 -*-
import os
import unittest

from mtg_nn_bot.draft_controller import DraftController


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = DraftController(os.path.expanduser("~/MTG"), "IXA_IXA_IXA_index.txt")


    def test1(self):
        draft1 = self.processor.start_draft([False,True,True,True,True,True,True,True], 1)
        draft2 = self.processor.start_draft([True,False,True,True,True,False,True,True], 2)
        self.processor.predict([draft1, draft2])
        self.processor.predict([draft1, draft2])
        self.processor.predict([draft1, draft2])
        self.log_draft(draft1, "/home/ivodopyanov/MTG/processor_tests/draft1.txt")
        self.log_draft(draft2, "/home/ivodopyanov/MTG/processor_tests/draft2.txt")

    def log_draft(self, draft, path):
        f = open(path, "wt")
        for pack_num in range(3):
            current_packs = [list(pack) for pack in draft['packs'][pack_num]]
            for pick_num in range(self.processor.model.settings['pack_size']):
                card_names = []
                for player_num in range(8):
                    chosen_card = draft['picks'][pack_num][player_num][pick_num]
                    if current_packs[player_num][chosen_card] == 0:
                        raise Exception("Pack {} pick {} player {}".format(pack_num, pick_num, player_num))
                    card_names.append(self.processor.index[current_packs[player_num][chosen_card]-1])
                    current_packs[player_num][chosen_card] = 0
                f.write("{}\n".format("||".join(card_names)))
                if pack_num%2==0:
                    current_packs = current_packs[-1:]+current_packs[:-1]
                else:
                    current_packs = current_packs[1:]+current_packs[:1]
        f.close()



if __name__ == "__main__":
    unittest.main()