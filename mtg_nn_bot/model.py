# -*- coding: utf-8 -*-
import tensorflow as tf
import json
import os
from io import open
import numpy as np
from . import DIR, INDEX, PACK_SIZE

class Model(object):
    def __init__(self, boosters):
        with open("../model_settings.json", "rt") as f:
            self.settings = json.load(f)
        self.boosters = boosters
        self.build_vocab()

    def build_vocab(self):
        self.vocab_decode = [0]
        self.vocab_encode = {0:0}
        expansions = sorted(set(self.boosters))
        for expansion in expansions:
            for card_number in INDEX.get_expansion(expansion).get_card_numbers():
                self.vocab_encode[card_number]= len(self.vocab_decode)
                self.vocab_decode.append(card_number)


    def build_trainer(self, card_embeddings, v, memory_layer, query_layer):
        packs = tf.placeholder(tf.int32, shape=[None, None], name="packs")
        picks = tf.placeholder(tf.int32, shape=[None, None], name="picks")

        this_pick = tf.placeholder(tf.int32, shape=[self.settings['batch_size'],], name="this_pick")

        dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")
        lr = tf.placeholder(dtype=tf.float32, shape=[], name="lr")


        packs_embedded = tf.nn.embedding_lookup(card_embeddings, packs, name="packs_embeddings")
        picks_embedded = tf.nn.embedding_lookup(card_embeddings, picks, name="picks_embeddings")

        picks_mask = tf.cast(tf.not_equal(picks, 0), tf.float32, name="picks_mask")
        picks_num = tf.reduce_sum(picks_mask, axis=1, name="reduced_picks_mask")
        picks_num = tf.where(tf.equal(picks_num, 0), tf.zeros_like(picks_num), 1 / picks_num)
        picks_weights = tf.multiply(tf.transpose(picks_mask, [1,0]), picks_num)
        picks_weighted = tf.transpose(tf.multiply(tf.transpose(picks_embedded, [2,1,0]), picks_weights), [2,1,0])
        picks_encoded = tf.reduce_sum(picks_weighted, axis=1)


        scores = tf.reduce_sum(v * tf.nn.tanh(memory_layer(packs_embedded) + tf.expand_dims(query_layer(picks_encoded), 1)), [2], name="scores")
        scores = tf.where(tf.equal(packs, 0), tf.zeros_like(scores), scores)
        scores_pred = tf.cast(tf.argmax(scores, axis=-1), tf.int32, name="scores_pred")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores,labels=this_pick)
        loss = tf.reduce_mean(loss, name="loss")

        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, name="train_op")

    def build_draft_predictor(self, card_embeddings, v, memory_layer, query_layer):
        with tf.variable_scope("draft"):
            packs = tf.placeholder(tf.int32, shape=[None, None, PACK_SIZE], name="packs")
            picked = tf.placeholder(tf.float32, shape=[None, None, self.settings['emb_dim']], name="picked")
            pack_num = tf.placeholder(tf.int32, shape=[None], name="pack_num")
            picks = tf.placeholder(tf.int32, shape=[None, None, PACK_SIZE], name="picks")

            packs_embedded = tf.nn.embedding_lookup(card_embeddings, packs, name="packs_embeddings")
            initial_packs_mask = tf.tile(tf.expand_dims(packs, -1), [1,1,1,self.settings['emb_dim']])
            packs_embedded = tf.where(tf.equal(initial_packs_mask, 0), tf.zeros_like(packs_embedded), packs_embedded)
            def pick_step(initializer, elems):
                pick_num = elems[0]
                picks = elems[1]
                current_packs = initializer[0]
                picked = initializer[1]
                pack_num = initializer[4]

                pack_card_mask = tf.reduce_max(current_packs, axis=-1)
                any_cards_in_pack_mask = tf.reduce_max(pack_card_mask, axis=-1)
                scores = tf.reduce_sum(v * tf.nn.tanh(memory_layer(current_packs) + tf.expand_dims(query_layer(picked/tf.cast(pick_num, tf.float32)), 2)), [3])
                scores = tf.where(tf.equal(pack_card_mask, 0), tf.ones_like(scores)*(-100), scores)
                chosen_cards = tf.cast(tf.argmax(scores, axis=-1), tf.int32)
                chosen_cards = tf.where(tf.equal(any_cards_in_pack_mask, 0), tf.ones_like(chosen_cards)*(-1), chosen_cards)
                chosen_cards = tf.where(tf.equal(picks, 0), chosen_cards, picks-1)

                chosen_cards_buf = tf.expand_dims(tf.one_hot(chosen_cards, PACK_SIZE),3)
                picked = picked + tf.reduce_sum(current_packs*chosen_cards_buf, axis=2)
                current_packs = current_packs*(1-chosen_cards_buf)

                current_packs = tf.where(tf.equal(tf.mod(pack_num, 2), 0),
                                         tf.concat([current_packs[:,-1:], current_packs[:,:-1]], axis=1),
                                         tf.concat([current_packs[:,1:], current_packs[:,:1]], axis=1))

                return [current_packs, picked, scores, chosen_cards, pack_num]

            scores_init = tf.zeros_like(packs_embedded[:,:,:,0])
            chosen_cards_init = tf.zeros_like(packs[:,:,0])
            result = tf.scan(fn=pick_step,
                             elems=[tf.range(1, PACK_SIZE+1), tf.transpose(picks, [2,0,1])],
                             initializer=[packs_embedded,
                                          picked,
                                          scores_init,
                                          chosen_cards_init,
                                          pack_num])
            picked = tf.identity(result[1][-1], name="result_picked")
            scores = tf.transpose(result[2], [1,2,0,3], name="predicted_scores")
            chosen_cards = tf.transpose(result[3], [1,2,0], name="predicted_picks")

    def build_p1p1_predictor(self, card_embeddings, v, memory_layer, query_layer):
        with tf.variable_scope("p1p1"):
            all_cards = tf.ones([1, len(self.vocab_decode)], dtype=tf.int32)
            packs_embedded = tf.nn.embedding_lookup(card_embeddings, all_cards, name="packs_embeddings")
            picked = tf.zeros([1, self.settings['emb_dim']], dtype=tf.float32)
            scores = tf.reduce_sum(v * tf.nn.tanh(memory_layer(packs_embedded) + tf.expand_dims(query_layer(picked), 1)), [2], name="p1p1_scores")





    def build(self):
        card_embeddings = tf.get_variable(name="card_embeddings_var",
                                          dtype=tf.float32,
                                          shape=[INDEX.get_card_count(set(self.boosters))+1, self.settings['emb_dim']])
        v = tf.get_variable("attention_v", [self.settings['rnn_units']], dtype=tf.float32)
        query_layer = tf.layers.Dense(units=self.settings['rnn_units'], use_bias=False, dtype=tf.float32, name="query_layer")
        memory_layer = tf.layers.Dense(units=self.settings['rnn_units'], use_bias=False, dtype=tf.float32, name="memory_layer")

        self.build_trainer(card_embeddings, v, memory_layer, query_layer)
        self.build_draft_predictor(card_embeddings, v, memory_layer, query_layer)
        self.build_p1p1_predictor(card_embeddings, v, memory_layer, query_layer)




    def get_feed_dict_predict(self, sess, X):
        packs = []
        picks = []
        for sample_id, sample in enumerate(X):
            packs.append([self.vocab_encode[card_num] for card_num in sample['pack']])
            picks.append(sample['picks'])
        feed = {
            sess.graph.get_tensor_by_name("packs:0"): packs,
            sess.graph.get_tensor_by_name("picks:0"): picks,
            sess.graph.get_tensor_by_name("dropout:0"): 1.0
        }
        return feed

    def get_feed_dict_train(self, sess, X, Y):
        feed = self.get_feed_dict_predict(sess, X)
        feed[sess.graph.get_tensor_by_name("dropout:0")] = self.settings['dropout']
        feed[sess.graph.get_tensor_by_name("lr:0")] = self.settings['lr']
        feed[sess.graph.get_tensor_by_name("this_pick:0")] = Y
        return feed

    def train(self, sess, X, Y):
        fd = self.get_feed_dict_train(sess, X, Y)
        _, train_loss, scores_pred, scores = sess.run([sess.graph.get_operation_by_name("train_op"),
                                                       sess.graph.get_tensor_by_name("loss:0"),
                                                       sess.graph.get_tensor_by_name("scores_pred:0"),
                                                       sess.graph.get_tensor_by_name("scores:0")],
                                                      feed_dict=fd)
        return train_loss, scores_pred

    def predict(self, sess, X):
        fd = self.get_feed_dict_predict(sess, X)
        Y_pred, scores = sess.run([sess.graph.get_tensor_by_name("scores_pred:0"),
                                   sess.graph.get_tensor_by_name("scores:0")], feed_dict=fd)
        return Y_pred, scores


    def get_feed_dict_predict_draft(self, sess, X):
        packs = np.zeros((len(X), X[0].packs.shape[1], PACK_SIZE), dtype=np.int32)
        picked = np.zeros((len(X), X[0].packs.shape[1], self.settings['emb_dim']), dtype=np.float32)
        pack_nums = np.zeros((len(X),), dtype=np.int32)
        picks = np.zeros((len(X), X[0].packs.shape[1], PACK_SIZE))
        for draft_id, draft in enumerate(X):
            for player_num in range(draft.packs.shape[1]):
                for card_pos in range(PACK_SIZE):
                    packs[draft_id, player_num, card_pos] = self.vocab_encode[draft.packs[draft.pack_num, player_num, card_pos]]
            picked[draft_id] = draft.picked
            pack_nums[draft_id] = draft.pack_num
            picks[draft_id] = draft.picks[draft.pack_num]
        feed = {
            sess.graph.get_tensor_by_name("draft/packs:0"): packs,
            sess.graph.get_tensor_by_name("draft/picked:0"): picked,
            sess.graph.get_tensor_by_name("draft/pack_num:0"): pack_nums,
            sess.graph.get_tensor_by_name("draft/picks:0"): picks
        }
        return feed


    def predict_draft(self, sess, X):
        fd = self.get_feed_dict_predict_draft(sess, X)
        new_picked, Y_pred, scores = sess.run([sess.graph.get_tensor_by_name("draft/result_picked:0"),
                                               sess.graph.get_tensor_by_name("draft/predicted_picks:0"),
                                               sess.graph.get_tensor_by_name("draft/predicted_scores:0")], feed_dict=fd)
        return new_picked, Y_pred, scores

    def predict_p1p1_scores_and_card_embeddings(self, sess):
        scores = sess.run([sess.graph.get_tensor_by_name("p1p1/p1p1_scores:0"),
                           sess.graph.get_tensor_by_name("card_embeddings_var:0")])
        p1p1_scores_data = scores[0].tolist()
        card_embeddings_data = scores[1].tolist()
        p1p1_scores = dict()
        card_embeddings = dict()
        for card_pos, card_id in enumerate(self.vocab_decode):
            if card_pos == 0:
                continue
            p1p1_scores[card_id]=p1p1_scores_data[0][card_pos]
            card_embeddings[card_id] = card_embeddings_data[card_pos]
        return p1p1_scores, card_embeddings



def load(boosters, sess):
    saver = tf.train.import_meta_graph(os.path.join(DIR, "models", "_".join(boosters), "model.meta"))
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(DIR, "models", "_".join(boosters), "model"))
    model = Model(boosters=boosters)
    return model