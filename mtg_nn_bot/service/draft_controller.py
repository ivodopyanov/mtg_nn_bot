# -*- coding: utf-8 -*-
import os
from mtg_nn_bot.model import load
import tensorflow as tf
import numpy as np
from random import choice, randint
import json

from .. import DIR

class Draft(object):
    def __init__(self, id, format_code, players, pack_num, packs, picks, picked):
        self.id = id
        self.format_code = format_code
        self.players = players
        self.pack_num = pack_num
        self.packs = packs
        self.picks = picks
        self.picked = picked

    def iterate_over_packs_in_round(self, round_num):
        packs = np.copy(self.packs[round_num])
        for pick_num in range(self.picks.shape[2]):
            yield packs
            for player_num, player in enumerate(self.players):
                chosen_card = self.picks[round_num, player_num, pick_num] - 1
                if chosen_card == -1:
                    continue
                if packs[player_num, chosen_card] == 0:
                    raise Exception("Pack {} pick {} player {}".format(round_num, pick_num, player_num))
                packs[player_num, chosen_card] = 0
            if round_num%2==0:
                packs = np.concatenate([packs[-1:],packs[:-1]], axis=0)
            else:
                packs = np.concatenate([packs[1:],packs[:1]], axis=0)

    def get_last_pick_num_made_by_player(self, player):
        return np.argmin(np.not_equal(self.picks[self.pack_num, player], 0))

    def to_json(self):
        result = {'packs': self.packs.tolist(),
                  'pack_num': self.pack_num,
                  'picks': self.picks.tolist(),
                  'picked': self.picked.tolist(),
                  'id': self.id,
                  'players': self.players,
                  'format_code': self.format_code}
        return json.dumps(result)

def load_draft_from_json(json_string):
    data = json.loads(json_string)
    draft = Draft(id=data['id'],
                  format_code=data['format_code'],
                  players=data['players'],
                  pack_num=data['pack_num'],
                  packs=np.array(data['packs']),
                  picks=np.array(data['picks']),
                  picked=np.array(data['picked']))
    return draft


class DraftController(object):
    def __init__(self, format_code):
        self.sess = tf.Session()
        self.model = load(format_code, self.sess)
        self.index = self.load_card_index(format_code)
        self.format_code = format_code

    def load_card_index(self, format_code):
        with open(os.path.join(DIR, "models", format_code, "index.txt"), "rt") as f:
            result = [line.rstrip("\n")[6:] for line in f]
            return result

    def generate_packs(self, players):
        packs = np.zeros((self.model.settings['round_num'], len(players), self.model.settings['pack_size']), dtype=np.int32)
        for round in range(self.model.settings['round_num']):
            for player_pos in range(len(players)):
                for card_pos in range(self.model.settings['pack_size']):
                    packs[round, player_pos, card_pos] = randint(0, len(self.index)-1)+1
        return packs

    def start_draft(self, players, draft_id):
        draft = Draft(id=draft_id,
                      format_code=self.format_code,
                      players=players,
                      pack_num=0,
                      packs=self.generate_packs(players),
                      picks=np.zeros((self.model.settings['round_num'], len(players), self.model.settings['pack_size']), dtype=np.int32),
                      picked=np.zeros((len(players), self.model.settings['emb_dim']), dtype=np.float32))
        return draft

    def predict(self, drafts):
        new_picked, new_picks, scores = self.model.predict_draft(self.sess, drafts)
        for draft_num, draft in enumerate(drafts):
            self.update_draft_state(draft, new_picks[draft_num]+1, new_picked[draft_num])


    def update_draft_state(self, draft, new_picks, new_picked):
        old_picks = draft.picks[draft.pack_num]
        valid_picks = [True for _ in range(8)]
        for pick_num in range(self.model.settings['pack_size']):
            for player_num in range(len(draft.players)):
                if old_picks[player_num, pick_num]!=0:
                    new_picks[player_num, pick_num] = old_picks[player_num, pick_num]
                else:
                    if not draft.players[player_num]:
                        valid_picks[player_num] = False
                    if not valid_picks[player_num]:
                        new_picks[player_num, pick_num] = 0
            if draft.pack_num%2==0:
                valid_picks = valid_picks[-1:]+valid_picks[:-1]
            else:
                valid_picks = valid_picks[1:]+valid_picks[:1]
        draft.picks[draft.pack_num] = new_picks
        if np.min(draft.picks[draft.pack_num, :, -1]) != 0:
            draft.picked = new_picked
            draft.pack_num += 1







