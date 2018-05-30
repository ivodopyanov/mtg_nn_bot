# -*- coding: utf-8 -*-
import os
from model import load, Model
import tensorflow as tf
import numpy as np
from random import choice, randint



def init_settings():
    settings = dict(emb_dim=32,
                    rnn_units=32,
                    batch_size=8,
                    max_card_count=1000,
                    dropout=.6,
                    lr=0.001,
                    dir=os.path.expanduser("~/MTG"),
                    progbar_label_count=3,
                    split=0.8,
                    pack_size=14)
    return settings

class DraftController(object):
    def __init__(self, model_path, index_path):
        self.sess = tf.Session()
        self.model = load(model_path, self.sess)
        #self.model = Model(init_settings())
        #self.model.build()
        #self.sess.run(tf.global_variables_initializer())
        self.index = self.load_card_index(os.path.join(model_path, index_path))

    def load_card_index(self, path):
        with open(path, "rt") as f:
            result = [line[6:-1] for line in f]
            return result

    def generate_packs(self):
        packs = []
        for _ in range(3):
            round_packs = []
            for _ in range(8):
                pack = []
                for _ in range(self.model.settings['pack_size']):
                    pack.append(randint(0, len(self.index)-1)+1)
                round_packs.append(pack)
            packs.append(round_packs)
        return packs

    def start_draft(self, bots, draft_id):
        draft_data = {'packs': self.generate_packs(),
                      'pack_num': 0,
                      'picks': np.zeros((3, 8, self.model.settings['pack_size']), dtype=np.int32),
                      'id': draft_id,
                      'picked': np.zeros((8, self.model.settings['emb_dim']), dtype=np.float32),
                      'bots': bots}
        return draft_data

    def predict(self, drafts):
        new_picked, pred, scores = self.model.predict_draft(self.sess, drafts)
        for draft_num, draft in enumerate(drafts):
            draft['picked'] = new_picked[draft_num]
            draft['picks'][draft['pack_num']] = pred[draft_num]
            draft['pack_num'] += 1



    def find_last_human_player_pick(self, picks, player_num):
        picks_by_player = [pick for pick in picks if picks['player']==player_num]
        picks_by_player.sort(key=lambda x:x['pick_num'])
        picks_by_player.sort(key=lambda x:x['pack_num'])
        return picks_by_player[-1]['pick_num']






