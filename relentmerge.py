from relentless import merge
from experiment_store import get_graph_from_a_folder
if __name__ == '__main__':
    merge(get_graph_from_a_folder('sample/basic'), get_graph_from_a_folder('/mnt/home/jkp/hack/tmp/MetaGPT/metagpt/ext/aflow/scripts/optimized/Zero/workflows/round_7'))