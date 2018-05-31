# -*- coding: utf-8 -*-
import os
from mtg_nn_bot.model import load, Model
import tensorflow as tf
import numpy as np
from random import choice, randint

class Draft(object):
    def __init__(self, packs, players, id, emb_dim):
        self.packs = packs
        self.pack_num = 0
        self.picks = np.zeros((packs.shape[0], len(players), packs.shape[2]), dtype=np.int32)
        self.picked = np.zeros((len(players), emb_dim), dtype=np.float32)
        self.id = id
        self.players = players

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


class DraftController(object):
    def __init__(self, model_path):
        self.sess = tf.Session()
        self.model = load(model_path, self.sess)
        self.index = self.load_card_index(os.path.join(model_path, "index.txt"))

    def load_card_index(self, path):
        with open(path, "rt") as f:
            result = [line[6:-1] for line in f]
            return result

    def generate_packs(self, players):
        packs = np.zeros((self.model.settings['round_num'], len(players), self.model.settings['pack_size']), dtype=np.int32)
        for round in range(self.model.settings['round_num']):
            for player_pos in range(len(players)):
                for card_pos in range(self.model.settings['pack_size']):
                    packs[round, player_pos, card_pos] = randint(0, len(self.index)-1)+1
        return packs

    def start_draft(self, players, draft_id):
        draft_data = Draft(packs=self.generate_packs(players),
                           players=players,
                           id=draft_id,
                           emb_dim=self.model.settings['emb_dim'])
        return draft_data

    def predict(self, drafts):
        new_picked, new_picks, scores = self.model.predict_draft(self.sess, drafts)
        for draft_num, draft in enumerate(drafts):
            draft_old_picks = draft.picks[draft.pack_num]
            draft_new_picks = new_picks[draft_num]+1
            valid_picks = [True for _ in range(8)]
            for pick_num in range(self.model.settings['pack_size']):
                for player_num in range(len(draft.players)):
                    if draft_old_picks[player_num, pick_num]!=0:
                        draft_new_picks[player_num, pick_num] = draft_old_picks[player_num, pick_num]
                    else:
                        if not draft.players[player_num]:
                            valid_picks[player_num] = False
                        if not valid_picks[player_num]:
                            draft_new_picks[player_num, pick_num] = 0
                if draft.pack_num%2==0:
                    valid_picks = valid_picks[-1:]+valid_picks[:-1]
                else:
                    valid_picks = valid_picks[1:]+valid_picks[:1]
            draft.picked = new_picked[draft_num]
            draft.picks[draft.pack_num] = draft_new_picks



    def find_last_human_player_pick(self, picks, player_num):
        picks_by_player = [pick for pick in picks if picks['player']==player_num]
        picks_by_player.sort(key=lambda x:x['pick_num'])
        picks_by_player.sort(key=lambda x:x['pack_num'])
        return picks_by_player[-1]['pick_num']






