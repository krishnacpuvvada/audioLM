import os

import sys
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_json_lines, write_json_lines
from utils import rround

import argparse

def parser_args():
    parser = argparse.ArgumentParser(description='cluster embeddings using faiss')
    parser.add_argument('--manifest_filepath_list', 
                        nargs='+', 
                        default=[], 
                        help='space separate listed of keys to extract from json',
                    )
    parser.add_argument('--master_manifest', type=str, required=True)
    parser.add_argument('--out_manifest_path', type=str, required=True)
    args = parser.parse_args()
    return args


def get_key_from_audio_filepath(audio_filepath):
    if 'voxceleb' in audio_filepath:
        key = audio_filepath.split('/')[-3:]
    
    if 'all_sre_wav' in audio_filepath:
        key = audio_filepath.split('/')[-2:]

    key = '_'.join(key)
    return key
    

def ref_manifest_to_dict(manifest_path):
    manifest_dict = {}
    for line in read_json_lines(manifest_path):
        audio_filepath = line['audio_filepath']
        key = get_key_from_audio_filepath(audio_filepath)
        manifest_dict[key] = line

    return manifest_dict


def main(args):
    ref_manifest_dict = ref_manifest_to_dict(args.master_manifest)
    print(f'len(ref_manifest_dict): {len(ref_manifest_dict)}')

    out_manifest_list = []
    for idx, manifest_path in enumerate(args.manifest_filepath_list):
        print(f'Processing file {manifest_path} {idx + 1}/{len(args.manifest_filepath_list)}')
        for line in read_json_lines(manifest_path):
            audio_filepath = line['audio_filepath']
            key = get_key_from_audio_filepath(audio_filepath)
            if key in ref_manifest_dict:
                out_manifest_list.append(line)

    print(f'len(out_manifest_list): {len(out_manifest_list)}')
    write_json_lines(args.out_manifest_path, out_manifest_list)    
    

if __name__=='__main__':
    args = parser_args()
    main(args)



