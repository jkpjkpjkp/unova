
import shelve

long_hash_to_short_hash = shelve.open('long_to_short.shelve')

import asyncio
import time

from db import find_the_strongest_graph, find_hardest_tasks
from graph import run_

if __name__ == "__main__":
    while True:
        print("Finding strongest graph and hardest tasks...")
        strongest_graph = find_the_strongest_graph()
        hardest_tasks_results = find_hardest_tasks(10) # This returns (Task, avg_correctness) tuples

        if strongest_graph and hardest_tasks_results:
            tasks_to_run = [task for task, _ in hardest_tasks_results] # Extract Task objects
            if tasks_to_run:
                print(f"Running {len(tasks_to_run)} hardest tasks on strongest graph...")
                try:
                    asyncio.run(asyncio.gather(*[run_(strongest_graph, task) for task in tasks_to_run]))
                    print("Run batch completed.")
                except Exception as e:
                    print(f"An error occurred during graph execution: {e}")
            else:
                print("No tasks extracted from hardest_tasks results.")
        else:
            print("Could not find strongest graph or hardest tasks. Waiting...")

        print("Sleeping for 10 seconds...")
        time.sleep(10) 