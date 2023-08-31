import argparse
import os
import numpy as np
import json
from tqdm import tqdm

import sys
sys.path.append("../")
from utils import read_json_lines, write_json_lines
from utils import rround
import numpy as np
from matplotlib import pyplot as plt

NUM_CODES = 8192

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path', type=str, default='/media/kpuvvada/data2/datasets/data/parallel_infer_ls960_CTs8/trial2/manifests/manifest_feature_all.json', help='comma separated list of manifest files')
    parser.add_argument('--key_to_load_from_manifest', type=str, default='audio_codes')
    args = parser.parse_args()
    return args


def codebook_utilization(code_counts, count_threshold):
    num_codes_used = 0
    for i in range(NUM_CODES):
        if code_counts[i] >= count_threshold:
            num_codes_used += 1

    return num_codes_used / NUM_CODES

def main(args):
    code_counts = np.array([0]* NUM_CODES)
    total_frames = 0

    lines = read_json_lines(args.manifest_path)
    for i, line in tqdm(enumerate(lines)):
        # print progress
        if i % 10000 == 0:
            print(f"Processing line {i} of {len(lines)}")

        # get codes for each file
        codes = np.load(line[args.key_to_load_from_manifest])['codes']
        for code in codes:
            code_counts[code] += 1
        total_frames += len(codes)

    for i in range(NUM_CODES):
        print(f"{i}: {code_counts[i]}")
    
    print(f"Total frames: {total_frames}")

    import pdb; pdb.set_trace()
    # stats
    min_count = np.min(code_counts)
    max_count = np.max(code_counts)
    mean_count = np.mean(code_counts)
    median_count = np.median(code_counts)

    uniform_count = 1/NUM_CODES * total_frames

    # put the above as dict
    utilization_thresholds = {'min_count': min_count,
                              'max_count': max_count,
                              'mean_count': mean_count,
                              'median_count': median_count,
                              'uniform_count': uniform_count,
                              'uniform_count/4': uniform_count/4,
                              'uniform_count/2': uniform_count/2,
                              '2* uniform_count': uniform_count* 2,
                              '4* uniform_count': uniform_count* 4,}
    
    
    # calculate utilization for different thresholds
    for th, th_val in utilization_thresholds.items():
        utilization = codebook_utilization(code_counts, th_val)
        # print
        print(f"Codebook utilization for {th}:{th_val} threshold: {rround(utilization, 4)}")

    # bar plot of code counts in log-10 scale and save to file
    # add axis labels
    plt.figure()
    plt.bar(np.arange(NUM_CODES), np.log10(code_counts))
    plt.xlabel('Code id')
    plt.ylabel('Log10 of code count')
    plt.savefig('code_counts.png')




if __name__=="__main__":
    args = parse_args()
    main(args)


