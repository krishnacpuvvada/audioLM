# reservoir sampling

import os
import sys
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import write_json_lines, read_json_lines, get_total_duration, rround
import random


def reservoir_sampling(stream, k, seed=None):
    """
    Reservoir sampling algorithm
    :param stream: list of items
    :param k: number of items to sample
    :param seed: random seed
    :return: list of sampled items
    """
    if seed is not None:
        random.seed(seed)
        
    sample = []
    for i, item in enumerate(stream):
        if i < k:
            sample.append(item)
        else:
            j = random.randint(0, i)
            if j < k:
                sample[j] = item
    return sample


def default_sampling(stream, k):
    return random.sample(stream, k)


def get_shard_audio_map(lines):
    shard_audio_map = {}
    for i, line in enumerate(lines):
        shard_id = int(line['shard_id'])
        if not shard_id in shard_audio_map.keys():
            shard_audio_map[shard_id] = []
        shard_audio_map[shard_id].append(line)

    return shard_audio_map, len(shard_audio_map)


def main():
    k = 15265           # librilight
    # k =  102750       # MMLPC en
    # k = 97300       # MMLPC es  
    # k = 77000       # MMLPC de
    # k = 87000          # MMLPC fr
    seed = 42

    random.seed(seed)

    input_manifest_path = "/home/kpuvvada/sub_sample_manifests/tarred_audio_manifest_LL.json"
    output_manifest_path = f"/home/kpuvvada/sub_sample_manifests/tarred_audio_manifest_LL_200hrs_seed{seed}.json"

    lines = read_json_lines(input_manifest_path)
    shard_audio_map, num_shards = get_shard_audio_map(lines)

    # print number of shards
    print(f'Number of shards: {num_shards}')
    samples_per_shard = k // num_shards
    print(f'Sampling {samples_per_shard} per shard ..')
    sample = []

    for shard_id, stream in shard_audio_map.items():
        sample.extend(default_sampling(stream, samples_per_shard))

    write_json_lines(output_manifest_path, sample)

    # calculate hrs
    total_duration = get_total_duration(output_manifest_path)

    # print in hrs
    print('Total duration: {} hrs'.format(rround(total_duration/3600)))


if __name__=="__main__":
    main()
