# -*- coding: utf-8 -*-
import os
from constants import *
from draft_controller import DraftController, load_draft_from_dict, Draft
from .. import DIR
from time import sleep
from collections import defaultdict
import datetime
import traceback

import json
import redis
import tensorflow as tf

R = redis.StrictRedis(host='localhost', port=6379, db=0)
DRAFT_CONTROLLERS = dict()
TIMEOUT = 1
STOP = False
SETTINGS = None


def run_server():
    while True:
        if STOP:
            break
        try:
            run_step()
        except Exception as e:
            with open(os.path.join(DIR, "logs", "{}.log".format(datetime.date.today().strftime('%d_%m_%Y'))), "at") as f:
                f.write("{} {}\n".format(datetime.datetime.now(), str(e)))
                f.write(traceback.format_exc())
            raise

        sleep(TIMEOUT)

def run_step():
    start_time = datetime.datetime.now()
    init()
    logfile = open(os.path.join(DIR, "logs", "{}.log".format(datetime.date.today().strftime('%d_%m_%Y'))), "at")
    drafts, new_drafts, training_data = get_drafts()
    logfile.write("{} - start\n".format(start_time))
    q = 1/0
    for format_code in drafts.keys():
        log_text = ["\t{}".format(format_code)]
        controller = DRAFT_CONTROLLERS[format_code]
        if controller.model.settings['training']==1:
            log_text.append("{}ms training".format((datetime.datetime.now()-start_time).microseconds/1000))
            training_data_buf = R.get(TRAINING_DATA_KEY.format(format_code))
            if training_data_buf is None:
                training_data_buf = {'X': [], 'Y': []}
            else:
                training_data_buf = json.loads(training_data_buf)
            if format_code in training_data:
                training_data_buf['X'].extend(training_data[format_code]['X'])
                training_data_buf['Y'].extend(training_data[format_code]['Y'])
            if len(training_data_buf['X']) >= controller.model.settings['batch_size']:
                loss, Y_pred = controller.online_training_step(training_data=training_data_buf)
                log_text.append("Loss = {:.2f}".format(loss))
                training_data_buf = {'X': [], 'Y': []}
            R.set(TRAINING_DATA_KEY.format(format_code), json.dumps(training_data_buf))
        log_text.append("{}ms predicting {} drafts".format((datetime.datetime.now()-start_time).microseconds/1000, len(drafts[format_code])))
        controller.predict(drafts[format_code])
        for draft in drafts[format_code]:
            if draft.pack_num != controller.model.settings['round_num']:
                R.set(DRAFT_KEY.format(draft.id), json.dumps(draft.to_dict()))
                R.sadd(READY_DRAFTS, draft.id)
                if draft.id in new_drafts:
                    R.set(DRAFT_TEMP_KEY.format(new_drafts[draft.id]), draft.id)
            else:
                R.delete(DRAFT_KEY.format(draft.id))
        log_text.append("{}ms end\n".format((datetime.datetime.now()-start_time).microseconds/1000))
        logfile.write(";\t".join(log_text))
    logfile.write("{} - end\n\n".format(datetime.datetime.now()))
    logfile.close()


def init():
    global SETTINGS
    for format_code in os.listdir(os.path.join(DIR, "models")):
        if format_code in DRAFT_CONTROLLERS.keys():
            continue
        controller = DraftController(format_code)
        DRAFT_CONTROLLERS[format_code] = controller
    if SETTINGS is None:
        if os.path.exists(os.path.join(DIR, "server_settings.json")):
            with open(os.path.join(DIR, "server_settings.json"), "rt") as f:
                SETTINGS = json.load(f)
        else:
            SETTINGS = {'draft_id': 1}
    else:
        with open(os.path.join(DIR, "server_settings.json"), "wt") as f:
            json.dump(SETTINGS, f)




def get_drafts():
    drafts = dict()
    training_data = dict()
    while True:
        make_pick_info = R.rpop(MAKE_PICK)
        if make_pick_info is None:
            break
        make_pick_info = json.loads(make_pick_info)
        if make_pick_info['id'] in drafts:
            draft = drafts[make_pick_info['id']]
        else:
            draft = load_draft_from_dict(json.loads(R.get(DRAFT_KEY.format(make_pick_info['id']))))
            drafts[make_pick_info['id']] = draft
        draft.picks[draft.pack_num, make_pick_info['player'], make_pick_info['pick_num']] = make_pick_info['pick_pos'] + 1

        if draft.format_code not in training_data:
             training_data[draft.format_code] = {'X': [], 'Y': []}
        pack = draft.get_pack(draft.pack_num, make_pick_info['player'], make_pick_info['pick_num'])
        picks = draft.get_picked(draft.pack_num, make_pick_info['player'], make_pick_info['pick_num'])
        training_data[draft.format_code]['X'].append({'pack': pack, 'picks': picks})
        training_data[draft.format_code]['Y'].append(make_pick_info['pick_pos'])

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
    return result, new_drafts, training_data

def get_free_draft_id():
    SETTINGS['draft_id'] += 1
    return SETTINGS['draft_id']


def stop_server():
    global STOP
    STOP = True

if __name__ == "__main__":
    run_server()