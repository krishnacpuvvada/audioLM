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
import copy
from tqdm import tqdm

import sys


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
    parser.add_argument('--save_path', type=str, default='/media/kpuvvada/data2/datasets/data/cached_models/descript/24khz/0.0.4/dac/codebooks.npz', help='save path for codes')
    args = parser.parse_args()
    return args


def get_model(tag="latest", model_type="24khz"):
    model = DAC()

    # Load compatible pre-trained model
    model = load_model(tag=tag, model_type=model_type)
    model.eval()
    return model


def main(args):
    emb_save_path = args.save_path

    # Load model
    model = get_model()
    codebooks = []

    with torch.no_grad():
        for module in model.quantizer.quantizers:
            emb_tensor = module.codebook.weight.detach()    # [n_codes, emb_dim]
            print("emb_tensor shape: ", emb_tensor.shape)
            emb_tensor = emb_tensor.unsqueeze(-1)           # [n_codes, emb_dim, 1]
            # conv 1D expects input as [N, C_in, L_in]  or [C_in, L_in]
            emb_tensor_out_proj = module.out_proj(emb_tensor)
            emb_tensor_out_proj = emb_tensor_out_proj.squeeze(-1) # [n_codes, emb_dim]
            codebooks.append(copy.deepcopy(emb_tensor_out_proj.cpu().numpy()))
    
    # concatenate codebooks
    codebooks = np.stack(codebooks, axis=0) # [n_codebooks, n_codes, emb_dim]
    # print shape
    print("codebooks shape: ", codebooks.shape)

    # save as npz file
    np.savez(emb_save_path, codebooks=codebooks)


if __name__ == '__main__':
    args = parse_args()
    main(args)