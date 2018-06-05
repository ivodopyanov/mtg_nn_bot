# -*- coding: utf-8 -*-
from .. import DIR
import os

if not os.path.exists(os.path.join(DIR, "drafts")):
    os.makedirs(os.path.join(DIR, "drafts"))
if not os.path.exists(os.path.join(DIR, "logs")):
    os.makedirs(os.path.join(DIR, "logs"))