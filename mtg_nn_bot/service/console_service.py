# -*- coding: utf-8 -*-
import os
import sys
import json

from mtg_nn_bot.service.draft_controller import DraftController


class ConsoleService(object):
    def __init__(self, format_code):
        self.processor = DraftController(format_code)

    def run(self):
        draft = self.processor.start_draft([False,True,True,True,True,True,True,True], 1)
        while True:
            current_pick_num = draft.get_last_pick_num_made_by_player(0)
            for pick_num, current_packs in enumerate(draft.iterate_over_packs_in_round(draft.pack_num)):
                if pick_num == current_pick_num:
                    break
            card_names = []
            for card_pos in range(self.processor.model.settings['pack_size']):
                if current_packs[0, card_pos]!=0:
                    card_names.append(u"{}) {}".format(card_pos, self.processor.index[current_packs[0, card_pos]-1]))
            sys.stdout.write("Pack {} Pick {}:\n{}\n".format(draft.pack_num, current_pick_num, u"\n".join(card_names)))
            chosen_card = int(raw_input("Input card #: "))
            draft.picks[draft.pack_num, 0, current_pick_num] = chosen_card+1
            self.processor.predict([draft])
            if current_pick_num == self.processor.model.settings['pack_size']-1:
                draft.pack_num += 1
            if draft.pack_num == self.processor.model.settings['round_num']:
                sys.stdout.write("Draft ended!\nNew draft started!\n")
                draft = self.processor.start_draft([False,True,True,True,True,True,True,True], 1)



if __name__ == "__main__":
    service = ConsoleService("IXA_IXA_IXA")
    service.run()