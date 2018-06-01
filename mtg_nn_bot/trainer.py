# -*- coding: utf-8 -*-
from model import Model
import os
import sys
import time

from random import shuffle
import tensorflow as tf
import json
from collections import defaultdict
from . import DIR

class Trainer(object):
    def __init___(self, format_code):
        self.format_code = format_code

    def load_data(self):
        with open(os.path.join(DIR, "models", self.format_code, "format_settings.json"), "rt") as f:
            self.settings = json.load(f)
        with open(os.path.join(DIR, "models", self.format_code, "index.txt"), "rt") as f:
            card_count = 0
            for _ in f:
                card_count += 1
            self.settings['card_count'] = card_count
        total_data = []
        with open(os.path.join(DIR, "models", self.format_code, "data.txt"), "rt") as f:
            for line_id, line in enumerate(f):
                line = line.replace("<",",")
                data = line.rstrip("\n").split(",")
                sample = [int(value) for value in data]
                total_data.append(sample)
        shuffle(total_data)
        return total_data

    def build_batch(self, batch):
        X = []
        Y = []
        for sample in batch:
            pack = sample[:(len(sample)-1)/4]
            picks = sample[(len(sample)-1)/4:-1]
            this_pick = sample[-1]
            shuffle(pack)
            X.append({'pack': pack, 'picks': picks})
            Y.append(pack.index(this_pick))
        return X, Y


    def train(self, epochs):
        data = self.load_data()
        split = int(len(data)*self.settings['split'])
        training_data = data[:split]
        test_data = data[split:]
        with tf.Session() as sess:
            model = Model(self.format_code, sess.graph)
            model.build()
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                sys.stdout.write("\nEpoch {}/{}\n".format(epoch+1, epochs))
                start_time = time.time()
                self.do_train(training_data, model, sess)
                self.do_test(test_data, model, sess)
                end_time = time.time()
                sys.stdout.write("Time for epoch: {:.2f}s\n".format((end_time-start_time)))
                tf.train.Saver(tf.trainable_variables()).save(sess, os.path.join(DIR, "models", self.format_code, "model"))



    def do_train(self, training_data, model, sess):
        sys.stdout.write("Training...\n")
        batch_num = len(training_data)/self.settings['batch_size']
        correct_preds = defaultdict(int)
        total_preds = defaultdict(int)
        total_correct = defaultdict(int)
        losses = []
        acc = []
        for batch in range(batch_num):
            X, Y = self.build_batch(training_data[self.settings['batch_size']*batch:self.settings['batch_size']*(batch+1)])
            loss, Y_pred = model.train(sess, X, Y)
            losses.append(loss)
            self.accumulate_preds(X, Y, Y_pred, correct_preds, total_preds, total_correct, acc)
            scores = self.calc_metrics(correct_preds, total_preds, total_correct)
            if batch%100 == 0:
                self.log_batch(batch, batch_num, scores, losses, acc)
        sys.stdout.write("\n")

    def do_test(self, test_data, model, sess):
        sys.stdout.write("Testing...\n")

        batch_num = len(test_data)/self.settings['batch_size']
        acc = []
        correct_preds = defaultdict(int)
        total_preds = defaultdict(int)
        total_correct = defaultdict(int)

        for batch in range(batch_num):
            X, Y = self.build_batch(test_data[self.settings['batch_size']*batch:self.settings['batch_size']*(batch+1)])
            Y_pred, _ = model.predict(sess, X)
            self.accumulate_preds(X, Y, Y_pred, correct_preds, total_preds, total_correct, acc)
        scores = self.calc_metrics(correct_preds, total_preds, total_correct)
        labels_by_score = sorted(scores.keys(), key=lambda label:scores[label])
        worst = labels_by_score[:self.settings['progbar_label_count']]
        text = []
        text.append("Accuracy: {:.2f}%".format(sum(acc)*100.0/len(acc)))
        for label_id, label in enumerate(worst):
            text.append("min score #{} ({}) = {:.2f}".format(label_id+1, label, 100*scores[label]))
        best = labels_by_score[-self.settings['progbar_label_count']:]
        for label_id, label in enumerate(best):
            text.append("max score #{} ({}) = {:.2f}".format(label_id+1, label, 100*scores[label]))

        text = "; ".join(text)
        sys.stdout.write(text+"\n")

    def accumulate_preds(self, X, Y_true, Y_pred, correct_preds, total_preds, total_correct, acc):
        for sample_id, pick in enumerate(Y_true):
            true_card_id = X[sample_id]['pack'][pick]
            pred_card_id = X[sample_id]['pack'][Y_pred[sample_id]]
            total_correct[true_card_id] += 1
            total_preds[pred_card_id] += 1
            if true_card_id == pred_card_id:
                correct_preds[true_card_id] += 1
            acc.append(true_card_id==pred_card_id)

    def calc_metrics(self, correct_preds, total_preds, total_correct):
        scores = defaultdict(int)
        for label in set(correct_preds.keys()+total_preds.keys()+total_correct.keys()):
            p = 1.0*correct_preds[label] / total_preds[label] if correct_preds[label] > 0 else 0
            r = 1.0*correct_preds[label] / total_correct[label] if correct_preds[label] > 0 else 0
            f1 = 2.0 * p * r / (p + r) if correct_preds[label] > 0 else 0
            scores[label] = f1
        if 0 in scores:
            del scores[0]
        return scores

    def log_batch(self, batch, batch_num, scores, losses, acc):
        labels_by_score = sorted(scores.keys(), key=lambda label:scores[label])
        worst = labels_by_score[:self.settings['progbar_label_count']]
        text = []
        text.append("Accuracy: {:.2f}%".format(sum(acc)*100.0/len(acc)))
        for label_id, label in enumerate(worst):
            text.append("min score #{} ({}) = {:.2f}".format(label_id+1, label, 100*scores[label]))
        best = labels_by_score[-self.settings['progbar_label_count']:]
        for label_id, label in enumerate(best):
            text.append("max score #{} ({}) = {:.2f}".format(label_id+1, label, 100*scores[label]))
        text = "; ".join(text)
        sys.stdout.write("\rBatch {}/{}: loss {:.4f}, {}".format(batch, batch_num, sum(losses)/len(losses), text))


if __name__ == "__main__":
    trainer = Trainer("IXA_IXA_IXA")
    trainer.train(5000)