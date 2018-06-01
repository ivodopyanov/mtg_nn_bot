# -*- coding: utf-8 -*-
import os
import unittest
import requests
import json
import threading

from ..service.rest_client import run_client
from ..service.rest_server import run_server, stop_server



URL_START_DRAFT = "http://127.0.0.1:5000/start_draft"
URL_MAKE_PICK = "http://127.0.0.1:5000/make_pick"
URL_SHUTDOWN = "http://127.0.0.1:5000/shutdown_mvutnsifhny"

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = threading.Thread(target=run_client)
        cls.server = threading.Thread(target=run_server)
        cls.client.start()
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        requests.post(URL_SHUTDOWN)
        stop_server()

    def test_single_player(self):
        response = requests.post(URL_START_DRAFT, {'format':"IXA_IXA_IXA", "players":[False,True,True,True,True,True,True,True]})
        result = json.loads(response.text)
        for pack_num in range(3):
            for player_num in range(8):
                if pack_num == 0:
                    self.assertEqual(player_num, result['picks'][pack_num][player_num].index(0))
                else:
                    self.assertEqual(0, result['picks'][pack_num][player_num].index(0))
        a = 1




if __name__ == "__main__":
    unittest.main()