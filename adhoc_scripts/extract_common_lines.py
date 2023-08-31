import os

import sys
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_json_lines, write_json_lines, read_lines, write_lines
from utils import rround

import argparse


def main():
    filename1 = "/home/kpuvvada/speaker_reco_manifests/train.json"
    filename2 = "/home/kpuvvada/speaker_reco_manifests/sre_all.json"

    out_manifest_path = "/home/kpuvvada/speaker_reco_manifests/sre_train.json"

    lines1 = set(read_lines(filename1))
    print(f'len(lines1): {len(lines1)}')

    lines2 = read_lines(filename2)
    lines2 = [x.replace("/data/all_sre_wav/", "/data/SpeakerRecognition/all_sre_wav/") for x in lines2]
    lines2 = set(lines2)

    common_lines = lines1.intersection(lines2)
    print(f'len(common_lines): {len(common_lines)}')
    write_lines(out_manifest_path, common_lines)

if __name__=='__main__':
    main()
