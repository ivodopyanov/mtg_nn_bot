# -*- coding: utf-8 -*-
import os
from constants import *
from draft_controller import DraftController, load_draft_from_json, Draft
from .. import DIR
from time import sleep
from collections import defaultdict

import json
import redis

R = redis.StrictRedis(host='localhost', port=6379, db=0)
DRAFT_CONTROLLERS = dict()
TIMEOUT = 1
STOP = False

def run_server():
    while True:
        if STOP:
            break
        init_models()
        drafts, new_drafts = get_drafts()
        for format_code in drafts.keys():
            controller = DRAFT_CONTROLLERS[format_code]
            controller.predict(drafts[format_code])
            for draft in drafts[format_code]:
                R.set(DRAFT_KEY.format(draft.id), draft.to_json())
                R.sadd(READY_DRAFTS, draft.id)
                if draft.id in new_drafts:
                    R.set(DRAFT_TEMP_KEY.format(new_drafts[draft.id]), draft.id)
        sleep(TIMEOUT)

def init_models():
    for format_code in os.listdir(os.path.join(DIR, "models")):
        if format_code in DRAFT_CONTROLLERS.keys():
            continue
        controller = DraftController(format_code)
        DRAFT_CONTROLLERS[format_code] = controller



def get_drafts():
    drafts = dict()
    while True:
        make_pick_info = R.rpop(MAKE_PICK)
        if make_pick_info is None:
            break
        make_pick_info = json.loads(make_pick_info)
        if make_pick_info['id'] in drafts:
            draft = drafts[make_pick_info['id']]
        else:
            draft = load_draft_from_json(R.get(DRAFT_KEY.format(make_pick_info['id'])))
            drafts[make_pick_info['id']] = draft
        draft.picks[draft.pack_num, make_pick_info['player'], make_pick_info['pick_num']] = make_pick_info['pick_pos']
    result = defaultdict(list)
    for draft in drafts.values():
        result[draft.format_code].append(draft)

    new_drafts = dict()
    while True:
        start_draft_info =R.rpop(START_DRAFT)
        if start_draft_info is None:
            break
        start_draft_info = json.loads(start_draft_info)
        draft = DRAFT_CONTROLLERS[start_draft_info['format']].start_draft(start_draft_info['players'], get_free_draft_id())
        result[draft.format_code].append(draft)
        new_drafts[draft.id] = start_draft_info['draft_temp_id']
    return result, new_drafts

def get_free_draft_id():
    for id in range(1,99999):
        if R.get(DRAFT_KEY.format(id)) is None:
            return id


def stop_server():
    global STOP
    STOP = True

if __name__ == "__main__":
    run_server()