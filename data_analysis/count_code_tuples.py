import argparse
import os
import numpy as np
import json
from tqdm import tqdm

import sys
sys.path.append("../")
from utils import read_json_lines, write_json_lines
from utils import rround


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path', type=str, help='comma separated list of manifest files')
    parser.add_argument('--key_to_load_from_manifest', type=str, default='audio_codes_filepath')
    args = parser.parse_args()
    return args

def convert_code_to_str(code, num_codebooks=2):
    code_str = ''
    for i in range(len(code[:num_codebooks])):
        code_str += f"{code[i]:04d}"

    return code_str

def return_code_tuple(fp):
    if fp.endswith('.npz'):
        codes = np.load(fp)['codes']

    code_str_count_dict = {}
    for i in range(codes.shape[1]):             # codes is [N, T]
        code_str = convert_code_to_str(codes[:, i])
        if code_str in code_str_count_dict:
            code_str_count_dict[code_str] += 1
        else:
            code_str_count_dict[code_str] = 1

    return code_str_count_dict, codes.shape[1]



def main(args):
    lines = read_json_lines(args.manifest_path)
    code_str_count_dict = {}
    total_frames = 0
    for line in tqdm(lines[:10000]):
        # get codes for each file
        tmp_dict, num_frames = return_code_tuple(line[args.key_to_load_from_manifest])

        # add to total frames
        total_frames += num_frames

        # add counts to code_str_count_dict
        for key in tmp_dict:
            if key in code_str_count_dict:
                code_str_count_dict[key] += tmp_dict[key]
            else:
                code_str_count_dict[key] = tmp_dict[key]

    # print number of keys
    print(f"Number of keys: {len(code_str_count_dict)}")

    # conver to tuple and sort by count
    code_str_count_list = []
    for key in code_str_count_dict:
        code_str_count_list.append((key, code_str_count_dict[key]))
    code_str_count_list.sort(key=lambda x: x[1], reverse=True)

    # make it cumulative
    code_str_count_list_cum = []
    cum_count = 0
    for key, count in code_str_count_list:
        cum_count += count
        code_str_count_list_cum.append((key, count, cum_count))

    # print top 10 as percentage of total frames
    for i in range(10):
        print(f"{code_str_count_list_cum[i][0]}: {rround(code_str_count_list_cum[i][2] / total_frames * 100)}%")

    # print every 10000th until 1000000 as percentage of total frames
    for i in range(10000, 100000, 10000):
        print(f"{code_str_count_list_cum[i][0]}: {rround(code_str_count_list_cum[i][2] / total_frames * 100)}%")




if __name__ == '__main__':
    args = parse_args()
    main(args)