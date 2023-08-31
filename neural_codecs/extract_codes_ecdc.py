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
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_json_lines, write_json_lines
from utils import rround


# encodec related imports
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset


class ListDataset(Dataset):
    def __init__(self, manifest_items, target_sr=24000, target_channels=1, dataset_name=None):
        self.manifest_items = manifest_items
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.manifest_items)
    
    def __getitem__(self, idx):
        item = self.manifest_items[idx]
        audio_filepath = item['audio_filepath']
        audio_filepath = self._maybe_correct_audio_filepath(audio_filepath)
        item['audio_filepath'] = audio_filepath   # update manifest

        wav, sr = torchaudio.load(audio_filepath) # [C, T]

        # account for offset and duration
        offset = float(item.get('offset', 0))
        duration = float(item.get('duration', -1))
        if duration > 0:
            wav = wav[:, int(offset * sr):int((offset + duration) * sr)]    # assuming duration does not exceed length of audio
        else:
            wav = wav[:, int(offset * sr):]

        wav = convert_audio(wav, sr, self.target_sr, self.target_channels)
        return item, wav.transpose(-1, -2)   # [C, T] -> [T, C]
    
    def _maybe_correct_audio_filepath(self, audio_filepath):
        if self.dataset_name == 'voxceleb':
            audio_filepath  = audio_filepath.replace('/disk4/manifests/', '/media/kpuvvada/data2/datasets/data/')

        if self.dataset_name == 'sre18':
            audio_filepath = audio_filepath.replace('/disk5/datasets/', '/data/')

        if self.dataset_name == 'sre_all':
            audio_filepath = audio_filepath.replace('/all_sre_wav/', '/all_sre_wav/')       # no change 

        return audio_filepath
    
    def _collate_fn(self, batch):
        items, audio, = zip(*batch)
        audio_lengths = torch.LongTensor([len(x) for x in audio])
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
        return items, audio.transpose(-1,-2), audio_lengths  # [B, T, C] -> [B, C, T]


def test_dataloader():
    lines = read_json_lines('/media/kpuvvada/data2/datasets/data/LibriSpeech/manifests/wavs/test_clean.json')
    ds = ListDataset(lines)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=ds._collate_fn)
    for names, audio, audio_lengths in dl:
        print(names)
        print(audio.shape)
        print(audio_lengths)
        break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path_list', type=str, help='comma separated list of manifest files')
    parser.add_argument('--out_manifest_path', type=str, default='codes.npy')
    parser.add_argument('--save_codes', action='store_true')
    parser.add_argument('--save_ecdc_wavs', action='store_true')
    parser.add_argument('--bit_rate', type=float, default=24.0)
    parser.add_argument('--speed_factor', type=float, default=1.0)
    parser.add_argument('--dataset', type=str, default='librispeech')
    parser.add_argument('--new_key', type=str, default='audio_codes_filepath')
    args = parser.parse_args()
    return args


def get_model(bandwith=6.0):
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(bandwith)
    model.eval()
    return model.cuda()


def get_codes(model, audio_filepath, speed_augmentor=None):
    duration = None # if speed augmentor is none, duration will be set to None
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    if speed_augmentor is not None:
        wav, _ = speed_augmentor(wav)
        duration = wav.shape[-1] / model.sample_rate
    wav = wav.unsqueeze(0)
    
    # extract codes
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
    codes = codes.squeeze(0)  # [n_q, T]
    codes = codes.detach().cpu().numpy()
    return codes, duration


def run_forward(model, audio_filepath):
    wav, sr = torchaudio.load(audio_filepath)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)  
    
    # run forward
    with torch.no_grad():
        estimated_wav = model.forward(wav)
    return estimated_wav


# torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)


def _maybe_correct_codes_filepath(args, codes_filepath, bitrate_directory_name, item=None):
    if args.dataset == 'voxceleb':
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('voxceleb_segments', f'voxceleb_segments_EnCodec/{bitrate_directory_name}_speed-{args.speed_factor}')
        else:
            codes_filepath = codes_filepath.replace('voxceleb_segments', f'voxceleb_segments_EnCodec/{bitrate_directory_name}')
    
    elif args.dataset == 'voxceleb-test':
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('voxceleb1', f'voxceleb1/EnCodec/{bitrate_directory_name}_speed-{args.speed_factor}')
        else:
            codes_filepath = codes_filepath.replace('voxceleb1', f'voxceleb1/EnCodec/{bitrate_directory_name}')

    elif args.dataset == 'sre18':
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('/sre18/eval/data/', f'/sre18/eval/EnCodec/{bitrate_directory_name}_speed-{args.speed_factor}/')
        else:
            codes_filepath = codes_filepath.replace('/sre18/eval/data/', f'/sre18/eval/EnCodec/{bitrate_directory_name}/')

    
    elif args.dataset == 'callhome':
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('/audio/', f'/EnCodec/{bitrate_directory_name}_speed-{args.speed_factor}/')
        else:
            codes_filepath = codes_filepath.replace('/audio/', f'/EnCodec/{bitrate_directory_name}/')

    elif args.dataset == 'spanish_mls':
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('/audio/', f'/EnCodec/{bitrate_directory_name}_speed-{args.speed_factor}/')
        else:
            codes_filepath = codes_filepath.replace('/test/wav/', f'/EnCodec/test_debug/{bitrate_directory_name}/')
    
    elif args.dataset == 'sre_all':
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('/all_sre_wav/', f'/all_sre_EnCodec/{bitrate_directory_name}_speed-{args.speed_factor}/')
        else:
            codes_filepath = codes_filepath.replace('/all_sre_wav/', f'/all_sre_EnCodec/{bitrate_directory_name}/')
        offset = rround(float(item.get('offset', 0)), 3)
        duration = rround(float(item.get('duration', -1)), 3)
        codes_filepath = codes_filepath.replace('.npz', f'_{offset}_{duration}.npz')
        
    else:
        if args.speed_factor != 1.0:
            codes_filepath = codes_filepath.replace('processed', f'encodec_codes/{bitrate_directory_name}_speed-{args.speed_factor}')
        else:
            codes_filepath = codes_filepath.replace('processed', f'encodec_codes/{bitrate_directory_name}')

    return codes_filepath
                     

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
    
    model = get_model(bandwith=bit_rate)

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
        manifest_items = read_json_lines(manifest_path)
        num_processed = 0
        if save_codes:
            predict_ds = ListDataset(manifest_items, dataset_name=args.dataset)
            predict_dl = DataLoader(predict_ds, batch_size=32, num_workers=20, shuffle=False, collate_fn=predict_ds._collate_fn, pin_memory=True)
            for items, audio, audio_lengths in tqdm(predict_dl):
                # extract codes
                with torch.no_grad():
                    encoded_frames = model.encode(audio.cuda())
                codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
                codes = codes.detach().cpu().numpy()

                # save codes in npz format        
                for idx, item in enumerate(items):
                    # round up length to nearest multiple of 75
                    l = int(np.ceil(audio_lengths[idx] / model.sample_rate * 75))
                    x = codes[idx, :, :l]

                    # save codes
                    audio_filepath = item['audio_filepath']
                    codes_filepath = audio_filepath.replace('.wav', '.npz')
                    codes_filepath = _maybe_correct_codes_filepath(args, codes_filepath, bitrate_directory_name, item)
                        
                    os.makedirs(os.path.dirname(codes_filepath), exist_ok=True)
                    np.savez(codes_filepath, codes=x)

                    num_processed += 1
                    if num_processed % 10000 == 0:
                        print(f'Processed {num_processed} files')
                    """
                    # update manifest
                    # codes_filepath = codes_filepath.replace('/media/kpuvvada/data2/datasets/data', '/data')
                    
                    item[args.new_key] = codes_filepath     
                    if duration is not None:
                        item['duration'] = rround(duration)    
                    out_manifest_items.append(item)
                    """
                    item[args.new_key] = codes_filepath
                    item['offset'] = 0.0
                    out_manifest_items.append(item)
            """
            if save_ecdc_wavs:
                target_sr = 16000
                target_encoding = 'PCM_S'
                target_bits_per_sample = 16
                comp_decomp_wav = run_forward(model, audio_filepath=audio_filepath)
                # assuming batchsize of one; remove zero dimension
                comp_decomp_wav = torch.squeeze(comp_decomp_wav, dim=0)

                # change sampling rate
                comp_decomp_wav = convert_audio(comp_decomp_wav, model.sample_rate, target_sr, 1)

                # save
                out_filepath = audio_filepath.replace('processed', f'encodec_wavs/{bitrate_directory_name}')
                os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
                torchaudio.save(out_filepath, comp_decomp_wav, sample_rate=target_sr, encoding='PCM_S', bits_per_sample=target_bits_per_sample)

                out_filepath = out_filepath.replace('/media/kpuvvada/data2/datasets/data', '/data')
                item['audio_filepath'] = out_filepath
                out_manifest_items.append(item)
            """

    # save manifest
    os.makedirs(os.path.dirname(out_manifest_path), exist_ok=True)
    write_json_lines(out_manifest_path, out_manifest_items)
        



if __name__ == '__main__':
    args = parse_args()
    main(args)

    # test_dataloader()