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
                # Assuming run_ expects a Graph object and a Task object
                # We need to make sure run_ is async if we use asyncio.gather
                # If run_ is not async, asyncio.gather won't work as expected.
                # Let's assume run_ is async for now.
                try:
                    # If run_ is async:
                    asyncio.run(asyncio.gather(*[run_(strongest_graph, task) for task in tasks_to_run]))
                    # If run_ is synchronous:
                    # for task in tasks_to_run:
                    #     run_(strongest_graph, task)
                    print("Run batch completed.")
                except Exception as e:
                    print(f"An error occurred during graph execution: {e}")
            else:
                print("No tasks extracted from hardest_tasks results.")
        else:
            print("Could not find strongest graph or hardest tasks. Waiting...")

        print("Sleeping for 10 seconds...")
        time.sleep(10) 