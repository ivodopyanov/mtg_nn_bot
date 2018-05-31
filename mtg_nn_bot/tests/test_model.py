# -*- coding: utf-8 -*-
import unittest
import tensorflow as tf
import os
import sys
from random import shuffle

from ..model import load

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sess = tf.Session()
        cls.model = load(os.path.expanduser("~/MTG/IXA_IXA_IXA"), cls.sess)
        cls.drafts = cls.load_data(os.path.expanduser("~/MTG/IXA_IXA_IXA/data.txt"))
        cls.index = cls.load_index(os.path.expanduser("~/MTG/IXA_IXA_IXA/index.txt"))


    @classmethod
    def load_data(cls, path):
        drafts = []
        draft = []
        with open(path, "rt") as f:
            for line_id, line in enumerate(f):
                line = line.replace("<",",")
                data = line.rstrip("\n").split(",")
                sample = [int(value) for value in data]
                if sample[(len(sample)-1)/4:-1].index(0) == 0 and len(draft) != 0:
                    drafts.append(draft)
                    draft = []
                draft.append(sample)
            drafts.append(draft)
        return drafts

    @classmethod
    def load_index(cls, path):
        index = []
        with open(path, "rt") as f:
            for line in f:
                line = line[6:-1]
                index.append(line)
        return index


    @classmethod
    def build_batch(self, draft_id):
        X = []
        Y = []
        for sample in self.drafts[draft_id]:
            pack = sample[:(len(sample)-1)/4]
            picks = sample[(len(sample)-1)/4:-1]
            this_pick = sample[-1]
            shuffle(pack)
            X.append({'pack': pack, 'picks': picks})
            Y.append(sample[-1])
        return X, Y

    def test1(self):
        for batch_num in range(10):
            with open(os.path.expanduser("~/MTG/tests/log_{}.txt".format(batch_num+1)), "wt") as f:
                X, Y = self.build_batch(batch_num)
                predicted, scores = self.model.predict(self.sess, X)
                for sample_id, sample in enumerate(X):
                    f.write("Pick {}:\n".format(sample_id+1))
                    card_texts = []
                    for card_num, card_id in enumerate(sample['pack']):
                        if card_id != 0:
                            card_text = "{}\t\t{:.4f}".format(self.index[card_id-1], scores[sample_id][card_num])
                            if predicted[sample_id] == card_num:
                                card_text = "{} <-- comp pick".format(card_text)
                            if Y[sample_id] == card_id:
                                card_text = "{} <-- player pick".format(card_text)
                            card_texts.append(card_text)
                    card_texts.sort()
                    f.write("{}\n\n".format(u"\n".join(card_texts)))




if __name__ == "__main__":
    unittest.main()