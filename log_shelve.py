import shelve
from PIL import Image
import hashlib
from multiprocessing import Lock
from typing import Any, Iterable, Optional, Union, List
import base64
import binascii
import io
import re
import os
import argparse

log_shelve = shelve.open('log.shelve')
tot = len(log_shelve.keys())
tot_lock = Lock()

def put(log: dict):
    global tot
    with tot_lock:
        tot += 1
        local_tot = tot
    log_shelve[str(local_tot)] = log
    return local_tot

def get(log_id: int):
    return log_shelve[str(log_id)]
