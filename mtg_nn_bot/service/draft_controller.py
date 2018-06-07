# -*- coding: utf-8 -*-
import os
from ..model import load, Model
import tensorflow as tf
import numpy as np
from random import choice, randint
import json

from .. import DIR, INDEX, PACK_SIZE

class Draft(object):
    def __init__(self, id, boosters, players, pack_num, packs, picks, picked, scores):
        self.id = id
        self.boosters = boosters
        self.players = players
        self.pack_num = pack_num
        self.packs = packs
        self.picks = picks
        self.picked = picked
        self.scores = scores

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

    def get_pack(self, round_num, player, pick_num):
        if round_num%2==0:
            original_pack_pos = (player-pick_num)%len(self.players)
        else:
            original_pack_pos = (player+pick_num)%len(self.players)
        pack = list(self.packs[round_num, original_pack_pos])

        for pick_num in range(pick_num):
            if round_num%2==0:
                intermediate_pick_pos = (original_pack_pos+pick_num)%len(self.players)
            else:
                intermediate_pick_pos = (original_pack_pos-pick_num)%len(self.players)
            if self.picks[round_num, intermediate_pick_pos, pick_num] == 0:
                return None
            pack[self.picks[round_num, intermediate_pick_pos, pick_num]-1] = 0
        return pack

    def get_picked(self, round_num, player, pick_num):
        buf = []
        for round in range(round_num):
            buf.extend(self.picks[round, player].tolist())
        buf.extend(self.picks[round_num, player,:pick_num].tolist())
        buf.extend([0]*(3*self.picks.shape[2]-len(buf)))
        return buf

    def to_dict(self):
        result = {'packs': self.packs.tolist(),
                  'pack_num': self.pack_num,
                  'picks': self.picks.tolist(),
                  'picked': self.picked.tolist(),
                  'id': self.id,
                  'players': self.players,
                  'boosters': self.boosters,
                  'scores': self.scores.tolist()}
        return result

def load_draft_from_dict(data):
    draft = Draft(id=data['id'],
                  boosters=data['boosters'],
                  players=data['players'],
                  pack_num=data['pack_num'],
                  packs=np.array(data['packs']),
                  picks=np.array(data['picks']),
                  picked=np.array(data['picked']),
                  scores=np.array(data['scores']))
    return draft


class DraftController(object):
    def __init__(self, boosters):
        self.sess = tf.Session()
        self.boosters = boosters
        if not os.path.exists(os.path.join(DIR, "models", "_".join(boosters))):
            os.makedirs(os.path.join(DIR, "models", "_".join(boosters)))
        if os.path.exists(os.path.join(DIR, "models", "_".join(boosters), "model.meta")):
            self.model = load(boosters, self.sess)
        else:
            self.model = Model(boosters=boosters)
            self.model.build()
            self.sess.run(tf.global_variables_initializer())


    def generate_boosters(self, players):
        packs = np.zeros((len(self.boosters), len(players), PACK_SIZE), dtype=np.int32)
        for round, expansion in enumerate(self.boosters):
            for player_pos in range(len(players)):
                packs[round, player_pos] = np.asarray(INDEX.get_expansion(expansion).generate_booster())
        return packs


    def start_draft(self, players, draft_id):
        draft = Draft(id=draft_id,
                      boosters=self.boosters,
                      players=players,
                      pack_num=0,
                      packs=self.generate_boosters(players),
                      picks=np.zeros((len(self.boosters), len(players), PACK_SIZE), dtype=np.int32),
                      picked=np.zeros((len(players), self.model.settings['emb_dim']), dtype=np.float32),
                      scores=np.zeros((len(self.boosters), len(players), PACK_SIZE, PACK_SIZE), dtype=np.float32))
        return draft

    def predict(self, drafts):
        new_picked, new_picks, scores = self.model.predict_draft(self.sess, drafts)
        for draft_num, draft in enumerate(drafts):
            self.update_draft_state(draft, new_picks[draft_num]+1, new_picked[draft_num], scores[draft_num])


    def update_draft_state(self, draft, new_picks, new_picked, scores):
        old_picks = draft.picks[draft.pack_num]
        valid_picks = [True for _ in range(8)]
        for pick_num in range(PACK_SIZE):
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
        draft.scores[draft.pack_num] = scores
        if np.count_nonzero(draft.picks[draft.pack_num]) == np.count_nonzero(draft.packs[draft.pack_num]):
            draft.picked = new_picked
            draft.pack_num += 1
            if draft.pack_num == len(self.boosters):
                with open(os.path.join(DIR, "drafts", "{}.json".format(draft.id)), "wt") as f:
                    draft_json = draft.to_dict()
                    del draft_json['picked']
                    del draft_json['pack_num']
                    json.dump(draft_json, f)

    def online_training_step(self, training_data):
        loss, Y_pred = self.model.train(sess=self.sess,
                                        X=training_data['X'],
                                        Y=training_data['Y'])
        tf.train.Saver(tf.trainable_variables()).save(self.sess, os.path.join(DIR, "models", "_".join(self.boosters), "model"))
        return loss, Y_pred


    def predict_p1p1_scores_and_card_embeddings(self):
        return self.model.predict_p1p1_scores_and_card_embeddings(self.sess)
