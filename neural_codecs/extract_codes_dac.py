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
from utils import rround


# dac related imports
import dac
from dac.utils import load_model
from dac.model import DAC

from dac.utils.encode import process as encode
from dac.utils.decode import process as decode

from audiotools import AudioSignal
import torchaudio
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path_list', type=str, help='comma separated list of manifest files')
    parser.add_argument('--out_manifest_path', type=str, default='codes.npy')
    parser.add_argument('--save_codes', action='store_true')
    parser.add_argument('--save_ecdc_wavs', action='store_true')
    parser.add_argument('--bit_rate', type=float, default=6.0)
    parser.add_argument('--speed_factor', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='librispeech')
    args = parser.parse_args()
    return args


def get_model(tag="latest", model_type="24khz"):
    model = DAC()

    # Load compatible pre-trained model
    model = load_model(tag=tag, model_type=model_type)
    model.eval()
    model.to('cuda')
    return model


def get_codes(model, audio_filepath, speed_augmentor=None):
    duration = None # if speed augmentor is none, duration will be set to None
    if speed_augmentor is not None:
        raise NotImplementedError('speed augmentor not implemented for DAC')
    
    signal = AudioSignal(audio_filepath)

    # Encode audio signal
    encoded_out = encode(signal, 'cuda', model)
    codes = encoded_out['codes']
    codes = codes.astype('int')

    codes = codes.squeeze(0)  # [n_q, T]
    return codes, duration


def run_forward(model, audio_filepath):
    raise NotImplementedError('run_forward not implemented for DAC')

    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)  
    
    # run forward
    with torch.no_grad():
        estimated_wav = model.forward(wav)
    return estimated_wav


# torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)

def main(args):
    manifest_path_list = args.manifest_path_list.split(',')
    out_manifest_path = args.out_manifest_path

    # what to save
    save_codes = args.save_codes
    save_ecdc_wavs = args.save_ecdc_wavs

    # some params
    bit_rate = args.bit_rate
    if bit_rate == 6.0:
        bitrate_directory_name = '6k'
    elif bit_rate == 12.0:
        bitrate_directory_name = '12k'
    elif bit_rate == 24.0:
        bitrate_directory_name = '24k'
    else:
        raise ValueError(f'bit_rate {bit_rate} not supported')
    
    model = get_model(tag="latest", model_type="24khz")

    # speed perturbation
    if args.speed_factor != 1.0:
        speed_augmentor = torchaudio.transforms.SpeedPerturbation(
            orig_freq=model.sample_rate,
            factors=[args.speed_factor]
        )
    else:
        speed_augmentor = None

    out_manifest_items = []
    for manifest_path in manifest_path_list:
        manifest = read_json_lines(manifest_path)
        for item in tqdm(manifest):
            audio_filepath = item['audio_filepath']
            
            if args.dataset == 'voxceleb':
                audio_filepath  = audio_filepath.replace('/disk4/manifests/', '/media/kpuvvada/data2/datasets/data/')
            
            if save_codes:
                codes, duration = get_codes(model, audio_filepath, speed_augmentor=speed_augmentor)

                # save codes in npz format
                codes_filepath = audio_filepath.replace('.wav', '.npz')

                if args.dataset == 'voxceleb':
                    if args.speed_factor != 1.0:
                        codes_filepath = codes_filepath.replace('voxceleb_segments', f'voxceleb_segments_dac/{bitrate_directory_name}_speed-{args.speed_factor}')
                    else:
                        codes_filepath = codes_filepath.replace('voxceleb_segments', f'voxceleb_segments_dac{bitrate_directory_name}')

                else:
                    if args.speed_factor != 1.0:
                        codes_filepath = codes_filepath.replace('processed', f'dac_codes/{bitrate_directory_name}_speed-{args.speed_factor}')
                    else:
                        codes_filepath = codes_filepath.replace('processed', f'dac_codes/{bitrate_directory_name}')

                    
                    
                os.makedirs(os.path.dirname(codes_filepath), exist_ok=True)
                np.savez(codes_filepath, codes=codes)

                # update manifest
                codes_filepath = codes_filepath.replace('/media/kpuvvada/data2/datasets/data', '/data')
                item['audio_codes_filepath'] = codes_filepath     
                if duration is not None:
                    item['duration'] = rround(duration)    
                out_manifest_items.append(item)


            if save_ecdc_wavs:
                raise NotImplementedError('save_ecdc_wavs not implemented for DAC')
                target_sr = 16000
                target_encoding = 'PCM_S'
                target_bits_per_sample = 16
                comp_decomp_wav = run_forward(model, audio_filepath=audio_filepath)
                # assuming batchsize of one; remove zero dimension
                comp_decomp_wav = torch.squeeze(comp_decomp_wav, dim=0)

                # change sampling rate
                comp_decomp_wav = convert_audio(comp_decomp_wav, model.sample_rate, target_sr, 1)

                # save
                out_filepath = audio_filepath.replace('processed', f'dac_wavs/{bitrate_directory_name}')
                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
                torchaudio.save(out_filepath, comp_decomp_wav, sample_rate=target_sr, encoding='PCM_S', bits_per_sample=target_bits_per_sample)

                out_filepath = out_filepath.replace('/media/kpuvvada/data2/datasets/data', '/data')
                item['audio_filepath'] = out_filepath
                out_manifest_items.append(item)

    # save manifest
    os.makedirs(os.path.dirname(out_manifest_path), exist_ok=True)
    write_json_lines(out_manifest_path, out_manifest_items)
        



if __name__ == '__main__':
    args = parse_args()
    main(args)