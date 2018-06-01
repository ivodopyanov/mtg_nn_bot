# -*- coding: utf-8 -*-
import sys
import os

from flask import Flask, request, render_template, jsonify
from flask_cache import Cache
from logging.handlers import RotatingFileHandler
import logging
import numpy as np
import redis
from time import sleep
import json

from constants import *
from draft_controller import load_draft_from_json, Draft

TIMEOUT = 0.5

app = Flask(__name__)
R = redis.StrictRedis()


@app.route("/start_draft", methods=['POST'])
def start_draft():
    data = dict(request.form)
    temp_id = get_free_temp_draft_id()
    data['draft_temp_id'] = temp_id
    data['format'] = data['format'][0]
    data['players'] = [value=="True" for value in data['players']]
    R.set(DRAFT_TEMP_KEY.format(temp_id), -1)
    R.rpush(START_DRAFT, json.dumps(data))
    while True:
        sleep(TIMEOUT)
        draft_id = R.get(DRAFT_TEMP_KEY.format(temp_id))
        if draft_id != "-1":
            R.delete(DRAFT_TEMP_KEY.format(temp_id))
            draft = json.loads(R.get(DRAFT_KEY.format(draft_id)))
            del draft['picked']
            return jsonify(draft)

@app.route("/make_pick", methods=['POST'])
def make_pick():
    draft_id = request.form['id']
    R.rpush(MAKE_PICK, json.dumps(dict(request.form)))
    while True:
        sleep(TIMEOUT)
        if R.sismember(READY_DRAFTS, draft_id):
            R.srem(READY_DRAFTS, draft_id)
            draft = json.loads(R.get(DRAFT_KEY.format(draft_id)))
            del draft['picked']
            return jsonify(draft)

@app.route('/shutdown_mvutnsifhny', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    func()
    return None

def get_free_temp_draft_id():
    for id in range(1,99999):
        if R.get(DRAFT_TEMP_KEY.format(id)) is None:
            return id




def run_client():
    logging.basicConfig(level=logging.INFO)
    handler = RotatingFileHandler("/tmp/ml_service.log", maxBytes=10*1024*1024, backupCount=10)
    formatter = logging.Formatter(fmt=u'%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    run_client()
