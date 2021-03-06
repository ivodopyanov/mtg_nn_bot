# -*- coding: utf-8 -*-
import sys
import os

from flask import Flask, request, render_template, jsonify
from logging.handlers import RotatingFileHandler
import logging
import numpy as np
import redis
from time import sleep
import json

from constants import *
from .. import DIR, SERVICE_TIMEOUT, REDIS_PORT, FLASK_PORT

app = Flask(__name__)
R = redis.StrictRedis(host='localhost', port=REDIS_PORT, db=0)


@app.route("/api/start_draft", methods=['POST'])
def start_draft():
    data = dict(request.form)
    if 'format' not in data:
        raise Exception("Missing parameter 'format'")
    if 'players' not in data:
        raise Exception("Missing parameter 'players'")
    temp_id = get_free_temp_draft_id()
    data['draft_temp_id'] = temp_id
    data['format'] = data['format'][0]
    data['players'] = [value=="True" for value in data['players']]
    R.set(DRAFT_TEMP_KEY.format(temp_id), -1)
    R.rpush(START_DRAFT, json.dumps(data))
    while True:
        sleep(SERVICE_TIMEOUT)
        draft_id = R.get(DRAFT_TEMP_KEY.format(temp_id))
        if draft_id != "-1":
            R.delete(DRAFT_TEMP_KEY.format(temp_id))
            R.srem(READY_DRAFTS, draft_id)
            draft = json.loads(R.get(DRAFT_KEY.format(draft_id)))
            del draft['picked']
            return jsonify(draft)

@app.route("/api/make_pick", methods=['POST'])
def make_pick():
    data = dict(request.form)
    if 'id' not in data:
        raise Exception("Missing parameter 'id'")
    if 'player' not in data:
        raise Exception("Missing parameter 'player'")
    if 'pick_num' not in data:
        raise Exception("Missing parameter 'pick_num'")
    if 'pick_pos' not in data:
        raise Exception("Missing parameter 'pick_pos'")
    data['id'] = int(data['id'][0])
    data['pick_num'] = int(data['pick_num'][0])
    data['player'] = int(data['player'][0])
    data['pick_pos'] = int(data['pick_pos'][0])
    R.rpush(MAKE_PICK, json.dumps(data))
    while True:
        sleep(SERVICE_TIMEOUT)
        if R.sismember(READY_DRAFTS, data['id']):
            R.srem(READY_DRAFTS, data['id'])
            draft = json.loads(R.get(DRAFT_KEY.format(data['id'])))
            del draft['picked']
            return jsonify(draft)
        elif os.path.exists(os.path.join(DIR, "drafts", "{}.json".format(data['id']))):
            with open(os.path.join(DIR, "drafts", "{}.json".format(data['id'])), "rt") as f:
                draft = json.load(f)
                return jsonify(draft)
        elif R.get(DRAFT_KEY.format(data['id'])) is None:
            raise Exception("Draft #{} is missing".format(data['id']))



@app.route("/api/get_draft", methods=['POST'])
def get_draft():
    if 'id' not in request.form:
        raise Exception("Missing parameter 'id'")
    id = request.form['id']
    draft = R.get(DRAFT_KEY.format(id))
    if draft is not None:
        draft = json.loads(draft)
        del draft['picked']
    elif os.path.exists(os.path.join(DIR, "drafts", "{}.json".format(id))):
        with open(os.path.join(DIR, "drafts", "{}.json".format(id)), "rt") as f:
            draft = json.load(f)
    else:
        raise Exception("Draft #{} is missing".format(id))
    return jsonify(draft)


@app.route("/api/p1p1", methods=['POST'])
def get_p1p1_scores():
    if 'format' not in request.form:
        raise Exception("Missing parameter 'format'")
    format = request.form['format']
    data = R.get(P1P1_SCORES.format(format))
    if data is None:
        data = {}
    else:
        data = json.loads(data)
    return jsonify(data)

@app.route("/api/card_embeddings", methods=['POST'])
def get_card_embeddings():
    if 'format' not in request.form:
        raise Exception("Missing parameter 'format'")
    format = request.form['format']
    data = R.get(CARD_EMBEDDINGS.format(format))
    if data is None:
        data = {}
    else:
        data = json.loads(data)
    return jsonify(data)




@app.route('/api/shutdown_mvutnsifhny', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    func()
    return jsonify("success")

def get_free_temp_draft_id():
    for id in range(1,99999):
        if R.get(DRAFT_TEMP_KEY.format(id)) is None:
            return id




def run_client():
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler("/tmp/ml_service2.log", maxBytes=10*1024*1024, backupCount=10)
    formatter = logging.Formatter(fmt=u'%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port=FLASK_PORT)

if __name__ == "__main__":
    run_client()
