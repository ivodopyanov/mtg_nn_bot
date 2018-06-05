# -*- coding: utf-8 -*-
from .. import DIR
import os

if not os.path.exists(os.path.join(DIR, "processor_tests")):
    os.makedirs(os.path.join(DIR, "processor_tests"))
if not os.path.exists(os.path.join(DIR, "tests")):
    os.makedirs(os.path.join(DIR, "tests"))
if not os.path.exists(os.path.join(DIR, "rest_tests")):
    os.makedirs(os.path.join(DIR, "rest_tests"))
