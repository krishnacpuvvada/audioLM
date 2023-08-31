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


def main():
    # librilight k = 15265
    # k =  102750       # MMLPC en
    # k = 97300       # MMLPC es  
    # k = 77000       # MMLPC de
    k = 87000          # MMLPC fr
    seed = 42

    input_manifest_path = "/home/kpuvvada/sub_sample_manifests/tarred_audio_manifest_pcstrip_fr.json"
    output_manifest_path = f"/home/kpuvvada/sub_sample_manifests/tarred_audio_manifest_pcstrip_fr_200hrs_seed{seed}.json"

    lines = read_json_lines(input_manifest_path)
    sample = reservoir_sampling(lines, k, seed)
    write_json_lines(output_manifest_path, sample)

    # calculate hrs
    total_duration = get_total_duration(output_manifest_path)

    # print in hrs
    print('Total duration: {} hrs'.format(rround(total_duration/3600)))


if __name__=="__main__":
    main()
