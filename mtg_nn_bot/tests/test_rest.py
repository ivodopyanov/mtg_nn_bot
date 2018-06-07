# -*- coding: utf-8 -*-
import os
import unittest
import requests
import json
import threading
import numpy as np
from random import choice
import redis
from time import sleep

from .. import DIR, INDEX
from ..service.rest_client import run_client
from ..service.rest_server import run_server, stop_server

R = redis.StrictRedis(host='localhost', port=6379, db=0)

URL_START_DRAFT = "http://127.0.0.1:5000/api/start_draft"
URL_MAKE_PICK = "http://127.0.0.1:5000/api/make_pick"
URL_GET_DRAFT = "http://127.0.0.1:5000/api/get_draft"
URL_P1P1_SCORES = "http://127.0.0.1:5000/api/p1p1"
URL_CARD_EMBEDDINGS = "http://127.0.0.1:5000/api/card_embeddings"
URL_SHUTDOWN = "http://127.0.0.1:5000/api/shutdown_mvutnsifhny"

#FORMAT = "XLN_XLN_XLN"
FORMAT = "ARB_DKA_MMQ"
PACK_SIZE = 15
ROUNDS = 3

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = threading.Thread(target=run_client)
        cls.server = threading.Thread(target=run_server)
        cls.client.start()
        cls.server.start()
        while R.spop("ready_drafts"):
            pass
        R.delete("training_data_{}".format(FORMAT))

    @classmethod
    def tearDownClass(cls):
        requests.post(URL_SHUTDOWN)
        stop_server()

    def test_p1p1_scores(self):
        response = requests.post(URL_P1P1_SCORES, {'format':FORMAT})
        result = json.loads(response.text)
        response = requests.post(URL_P1P1_SCORES, {'format':"XLN_RIX_XLN"})
        result = json.loads(response.text)
        self.assertEqual(result, {})

    def test_card_embeddings(self):
        response = requests.post(URL_CARD_EMBEDDINGS, {'format':FORMAT})
        result = json.loads(response.text)
        response = requests.post(URL_CARD_EMBEDDINGS, {'format':"XLN_RIX_XLN"})
        result = json.loads(response.text)
        self.assertEqual(result, {})


    def test_single_player(self):
        output_file = open(os.path.join(DIR, "rest_tests", "draft.txt"), "wt")
        response = requests.post(URL_START_DRAFT, {'format':FORMAT, "players":[False,True,True,True,True,True,True,True]})
        result = json.loads(response.text)
        for pack_num in range(ROUNDS):
            for player_num in range(8):
                if pack_num == 0:
                    self.assertEqual(player_num, result['picks'][pack_num][player_num].index(0))
                else:
                    self.assertEqual(0, result['picks'][pack_num][player_num].index(0))
        for pack_num in range(ROUNDS):
            current_packs = np.asarray(result['packs'][pack_num])
            expansion = FORMAT.split("_")[pack_num]
            for pick_num in range(PACK_SIZE):
                potential_card_pos = [i for i in range(current_packs.shape[1]) if current_packs[0][i]!=0]
                if len(potential_card_pos) == 0:
                    break
                card_pos = choice(potential_card_pos)
                response = requests.post(URL_MAKE_PICK, {'id': result['id'], 'player': 0, 'pick_num': pick_num, 'pick_pos': card_pos})
                result = json.loads(response.text)
                picks = np.asarray(result['picks'])
                for player_num in range(picks.shape[1]):
                    picked_card_pos = picks[pack_num][player_num][pick_num] - 1
                    card_names = []
                    for card_in_pack_pos in range(current_packs.shape[1]):
                        if current_packs[player_num][card_in_pack_pos]!=0:
                            card_name = INDEX.get_expansion(expansion).get_card_name(current_packs[player_num][card_in_pack_pos]-1)
                            if card_in_pack_pos == picked_card_pos:
                                card_name = "--> {} <--".format(card_name)
                            card_names.append(card_name)
                    output_file.write("Pack {} Pick {} Player {}:\n{}\n\n".format(pack_num, pick_num, player_num, "\t".join(card_names)))
                    if current_packs[player_num][picked_card_pos] == 0:
                        raise Exception("Error: pack {} player {} pick {} picked missing card # {}".format(pack_num, player_num, pick_num, picked_card_pos))
                    current_packs[player_num][picked_card_pos] = 0
                if pack_num%2==0:
                    current_packs = np.concatenate([current_packs[-1:],current_packs[:-1]], axis=0)
                else:
                    current_packs = np.concatenate([current_packs[1:],current_packs[:1]], axis=0)
        output_file.close()

    def test_multiplayer(self):
        response = requests.post(URL_START_DRAFT, {'format':FORMAT, "players":[False,True,True,False,False,True,True,True]})
        result = json.loads(response.text)
        player0_thread = threading.Thread(target=draft_player, args=(0, result))
        player3_thread = threading.Thread(target=draft_player, args=(3, result))
        player4_thread = threading.Thread(target=draft_player, args=(4, result))
        player0_thread.start()
        player3_thread.start()
        player4_thread.start()
        player0_thread.join()
        player3_thread.join()
        player4_thread.join()

        output_file = open(os.path.join(DIR, "rest_tests", "draft_multi.txt"), "wt")
        response = requests.post(URL_GET_DRAFT, {'id': result['id']})
        draft = json.loads(response.text)
        picks = np.asarray(draft['picks'])
        for pack_num in range(ROUNDS):
            current_packs = np.asarray(draft['packs'][pack_num])
            expansion = FORMAT.split("_")[pack_num]
            for pick_num in range(PACK_SIZE):
                for player_num in range(picks.shape[1]):
                    picked_card_pos = picks[pack_num][player_num][pick_num] - 1
                    if picked_card_pos == -1:
                        continue
                    card_names = []
                    for card_in_pack_pos in range(current_packs.shape[1]):
                        if current_packs[player_num][card_in_pack_pos]!=0:
                            card_name = INDEX.get_expansion(expansion).get_card_name(current_packs[player_num][card_in_pack_pos]-1)
                            if card_in_pack_pos == picked_card_pos:
                                card_name = "--> {} <--".format(card_name)
                            card_names.append(card_name)
                    output_file.write("Pack {} Pick {} Player {}:\n{}\n\n".format(pack_num, pick_num, player_num, "\t".join(card_names)))
                    if current_packs[player_num][picked_card_pos] == 0:
                        raise Exception("Error: pack {} player {} pick {} picked missing card # {}".format(pack_num, player_num, pick_num, picked_card_pos))
                    current_packs[player_num][picked_card_pos] = 0
                if pack_num%2==0:
                    current_packs = np.concatenate([current_packs[-1:],current_packs[:-1]], axis=0)
                else:
                    current_packs = np.concatenate([current_packs[1:],current_packs[:1]], axis=0)
        output_file.close()



def draft_player(player, draft):
    for pack_num in range(ROUNDS):
        for pick_num in range(PACK_SIZE):
            current_pack = get_current_pack(player, draft, pack_num)
            while current_pack is None:
                sleep(0.5)
                response = requests.post(URL_GET_DRAFT, {'id': draft['id']})
                draft = json.loads(response.text)
                current_pack = get_current_pack(player, draft, pack_num)
            potential_card_pos = [i for i in range(len(current_pack)) if current_pack[i]!=0]
            if len(potential_card_pos) == 0:
                break
            card_pos = choice(potential_card_pos)
            response = requests.post(URL_MAKE_PICK, {'id': draft['id'], 'player': player, 'pick_num': pick_num, 'pick_pos': card_pos})
            draft = json.loads(response.text)
        pack_not_ended = any(player_num for player_num in range(len(draft['players'])) if draft['picks'][pack_num][player_num][-1]==0)
        if pack_not_ended:
            sleep(0.5)
            response = requests.post(URL_GET_DRAFT, {'id': draft['id']})
            draft = json.loads(response.text)
            pack_not_ended = any(player_num for player_num in range(len(draft['players'])) if draft['picks'][pack_num][player_num][-1]==0)




def get_current_pack(player, draft, pack_num):
    first_zero_pick = draft['picks'][pack_num][player].index(0)
    if pack_num%2==0:
        original_pack_pos = (player-first_zero_pick)%len(draft['players'])
    else:
        original_pack_pos = (player+first_zero_pick)%len(draft['players'])
    pack = list(draft['packs'][pack_num][original_pack_pos])

    for pick_num in range(first_zero_pick):
        if pack_num%2==0:
            intermediate_pick_pos = (original_pack_pos+pick_num)%len(draft['players'])
        else:
            intermediate_pick_pos = (original_pack_pos-pick_num)%len(draft['players'])
        if draft['picks'][pack_num][intermediate_pick_pos][pick_num] == 0:
            return None
        pack[draft['picks'][pack_num][intermediate_pick_pos][pick_num]-1] = 0
    return pack



if __name__ == "__main__":
    unittest.main()