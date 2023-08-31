# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extracts the codes from the trained model and saves them in a npz file and upate manifest

import argparse
import os
import numpy as np
import json
from tqdm import tqdm

import sys
sys.path.append("../")
from utils import read_json_lines, write_json_lines


# encodec related imports
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path_list', type=str, help='comma separated list of manifest files')
    parser.add_argument('--out_manifest_path', type=str, default='codes.npy')
    args = parser.parse_args()
    return args


def main(args):
    manifest_path_list = args.manifest_path_list.split(',')
    out_manifest_path = args.out_manifest_path

    bit_rate = 6.0      # in kbps
    bitrate_directory_name  = '6k'
    
    out_manifest_items = []
    cmd_list = []
    for manifest_path in manifest_path_list:
        manifest = read_json_lines(manifest_path)
        for i, item in enumerate(tqdm(manifest)):
            audio_filepath = item['audio_filepath']

            # cmd to compress and de-compress audio
            # encodec [-r] [-b TARGET_BANDWIDTH] [-f] [--hq] [--lm] INPUT_FILE OUTPUT_WAV_FILE
            output_filepath = audio_filepath.replace('processed', f'encodec_wavs/{bitrate_directory_name}')
            cmd = f'encodec -b {bit_rate} -f {audio_filepath} {output_filepath}' 
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            # run system command
            # os.system(cmd)
            cmd_list.append(cmd)

            # update manifest
            output_filepath = output_filepath.replace('/media/kpuvvada/data2/datasets/data', '/data')
            item['audio_filepath'] = output_filepath     
            out_manifest_items.append(item)

    # save manifest
    write_json_lines(out_manifest_path, out_manifest_items)

    # run cmd_list in parallel using multiprocessing process
    # use tqdm to show progress
    import multiprocessing
    pool = multiprocessing.Pool(processes=4)
    for _ in tqdm(pool.imap_unordered(os.system, cmd_list[:100]), total=len(cmd_list[:100])):
        pass
    pool.close()
    pool.join()








if __name__ == '__main__':
    args = parse_args()
    main(args)