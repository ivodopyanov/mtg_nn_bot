# -*- coding: utf-8 -*-
import tensorflow as tf
import json
import os
from io import open
import numpy as np


class Model(object):
    def __init__(self, settings):
        self.settings = settings


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
            packs = tf.placeholder(tf.int32, shape=[None, self.settings['player_num'], self.settings['pack_size']], name="packs")
            picked = tf.placeholder(tf.float32, shape=[None, self.settings['player_num'], self.settings['emb_dim']], name="picked")
            pack_num = tf.placeholder(tf.int32, shape=[None], name="pack_num")
            picks = tf.placeholder(tf.int32, shape=[None, self.settings['player_num'], self.settings['pack_size']], name="picks")

            packs_embedded = tf.nn.embedding_lookup(card_embeddings, packs, name="packs_embeddings")
            def pick_step(initializer, elems):
                pick_num = elems[0]
                picks = elems[1]
                current_packs = initializer[0]
                picked = initializer[1]
                pack_num = initializer[4]

                pack_card_mask = tf.reduce_max(current_packs, axis=-1)
                scores = tf.reduce_sum(v * tf.nn.tanh(memory_layer(current_packs) + tf.expand_dims(query_layer(picked/tf.cast(pick_num, tf.float32)), 2)), [3])
                scores = tf.where(tf.equal(pack_card_mask, 0), tf.zeros_like(scores), scores)
                chosen_cards = tf.cast(tf.argmax(scores, axis=-1), tf.int32)
                chosen_cards = tf.where(tf.equal(picks, 0), chosen_cards, picks-1)

                chosen_cards_buf = tf.expand_dims(tf.one_hot(chosen_cards, self.settings['pack_size']),3)
                picked = picked + tf.reduce_sum(current_packs*chosen_cards_buf, axis=2)
                current_packs = current_packs*(1-chosen_cards_buf)

                current_packs = tf.where(tf.equal(tf.mod(pack_num, 2), 0),
                                         tf.concat([current_packs[:,-1:], current_packs[:,:-1]], axis=1),
                                         tf.concat([current_packs[:,1:], current_packs[:,:1]], axis=1))

                return [current_packs, picked, scores, chosen_cards, pack_num]

            scores_init = tf.zeros_like(packs_embedded[:,:,:,0])
            chosen_cards_init = tf.zeros_like(packs[:,:,0])
            result = tf.scan(fn=pick_step,
                             elems=[tf.range(1, self.settings['pack_size']+1), tf.transpose(picks, [2,0,1])],
                             initializer=[packs_embedded,
                                          picked,
                                          scores_init,
                                          chosen_cards_init,
                                          pack_num])
            picked = tf.identity(result[1][-1], name="result_picked")
            scores = tf.transpose(result[2], [1,2,0,3], name="predicted_scores")
            chosen_cards = tf.transpose(result[3], [1,2,0], name="predicted_picks")



    def build(self):
        card_embeddings = tf.get_variable(name="card_embeddings_var",
                                          dtype=tf.float32,
                                          shape=[self.settings['card_count']+1, self.settings['emb_dim']])
        v = tf.get_variable("attention_v", [self.settings['rnn_units']], dtype=tf.float32)
        query_layer = tf.layers.Dense(units=self.settings['rnn_units'], use_bias=False, dtype=tf.float32, name="query_layer")
        memory_layer = tf.layers.Dense(units=self.settings['rnn_units'], use_bias=False, dtype=tf.float32, name="memory_layer")

        self.build_trainer(card_embeddings, v, memory_layer, query_layer)
        self.build_draft_predictor(card_embeddings, v, memory_layer, query_layer)




    def get_feed_dict_predict(self, sess, X):
        packs = []
        picks = []
        for sample in X:
            packs.append(sample['pack'])
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
        packs = np.zeros((len(X), self.settings['player_num'], self.settings['pack_size']), dtype=np.int32)
        picked = np.zeros((len(X), self.settings['player_num'], self.settings['emb_dim']), dtype=np.float32)
        pack_nums = np.zeros((len(X),), dtype=np.int32)
        picks = np.zeros((len(X), self.settings['player_num'], self.settings['pack_size']))
        for draft_id, draft in enumerate(X):
            packs[draft_id] = draft.packs[draft.pack_num]
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



def load(path, sess):
    with open(os.path.join(path, "settings.json"), "rt", encoding="utf8") as f:
        settings = json.load(f)
        model = Model(settings=settings)
        saver = tf.train.import_meta_graph(os.path.join(path, "model.meta"), graph=sess.graph)
        saver.restore(sess, os.path.join(path, "model"))
        return model