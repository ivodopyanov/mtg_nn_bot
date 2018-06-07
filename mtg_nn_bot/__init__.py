# -*- coding: utf-8 -*-
import json
import os
import numpy as np

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
PACK_SIZE = 15
REDIS_TIMEOUT = 60*30
SERVICE_TIMEOUT = 0.5
PROBS = {"rare":7}

with open(os.path.join(__location__, "settings.json"), "rt") as f:
    settings = json.load(f)
    DIR = os.path.expanduser(settings['dir'])
    REDIS_PORT = settings['redis_port']
    FLASK_PORT = settings['flask_port']

class Expansion(object):
    def __init__(self, data):
        self.data = data

    def card_iterator(self):
        last_rarity = ""
        for card in self.data['cards']:
            if card.get('starter', False):
                continue
            if card['rarity'] != "Basic Land" and last_rarity == "Basic Land":
                break
            last_rarity = card['rarity']
            if 'mciNumber' in card:
                number = card['mciNumber']
                if number[-1].isalpha() and number[-1]!="a":
                    continue
            yield card

    def get_card_count(self):
        return len(self.get_card_numbers())

    def get_card_numbers(self):
        result = set()
        for card in self.card_iterator():
            result.add(card['multiverseid'])
        return result

    def get_cards_by_type(self, rarity):
        if rarity.startswith("foil"):
            rarity=rarity[len("foil "):]
        double_faced = rarity.startswith("double faced")
        if double_faced:
            rarity = rarity[len("double faced "):]

        result = set()
        for card in self.card_iterator():
            if len(rarity)!=0 and card['rarity'].lower() != rarity:
                continue
            if double_faced and not card["number"][-1].isalpha():
                continue
            result.add(card['multiverseid'])
        return result

    def generate_booster(self):
        result = []
        for card_pos, card_type in enumerate(self.data['booster']):
            if type(card_type) is list:
                probs = [PROBS.get(t, 1) for t in card_type]
                probs = [p*1.0/sum(probs) for p in probs]
                card_type = np.random.choice(card_type, p=probs)
            if card_type == "land":
                card_type = "basic land"
            card_numbers = self.get_cards_by_type(card_type)
            if len(card_numbers) != 0:
                card_number = np.random.choice(list(card_numbers))
                result.append(card_number)
        while len(result) < PACK_SIZE:
            result.append(0)
        return result

    def get_card_name(self, multiverseid):
        for card in self.data['cards']:
            if card['multiverseid'] == multiverseid:
                return card['name']
        return 'NONE'


class Index(object):
    def __init__(self):
        with open(os.path.join(DIR, "AllSets.json"), "rt") as f:
            self.data = json.load(f)

    def get_card_count(self, sets):
        return sum(Expansion(self.data[s]).get_card_count() for s in sets)

    def get_expansion(self, code):
        return Expansion(self.data[code])

    def get_draftable_expansions(self):
        result = [key for key in self.data.keys() if len(self.data[key].get('booster',[]))==15 or len(self.data[key].get('booster',[]))==16]
        result.remove("TSB")
        result.remove("TSP")
        result.remove("PLC")
        result.remove("FUT")
        result.remove("ME4")
        result.remove("CNS")
        return result



INDEX = Index()