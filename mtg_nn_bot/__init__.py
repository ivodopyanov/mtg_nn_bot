# -*- coding: utf-8 -*-
import json
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

with open(os.path.join(__location__, "settings.json"), "rt") as f:
    settings = json.load(f)
    DIR = os.path.expanduser(settings['dir'])