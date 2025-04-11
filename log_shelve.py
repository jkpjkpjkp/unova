import shelve
from multiprocessing import Lock

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
