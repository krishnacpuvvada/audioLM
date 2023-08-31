# just convert to nemo manifest format

import os
import sys
# append parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import write_json_lines, read_lines


out_filepath = './manifest.json'
input_filepath = '/media/kpuvvada/data2/datasets/data/LibriLight/pt_list'


new_manifest = []
lines = read_lines(input_filepath)

for line in lines:
    line = os.path.splitext(line)[0] + '.flac'
    # entry = {'audio_codes': f'/media/kpuvvada/data2/datasets/data/LibriLight/encodec_pt/{line}'}
    entry = {'audio_filepath': f'/mnt/drive1/librilight/flac/{line}', 'duration': 10000}

    new_manifest.append(entry)

write_json_lines(out_filepath, new_manifest)

# print current directory
print(os.getcwd())

