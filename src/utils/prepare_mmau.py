import os
import json
import argparse

parser = argparse.ArgumentParser(description='Prepare mmau dataset.')
parser.add_argument('--input_file', help="input json file", required=True)
parser.add_argument('--out_file', help="output format file", required=True)
parser.add_argument('--wav_dir', help="wav dir", required=True)
"""
Source: https://github.com/Sakshi113/mmau
"""

args = parser.parse_args()
input_file = args.input_file
out_file = args.out_file
wav_dir = args.wav_dir


def handle_mmau():
    with open(input_file, "r") as f:
        input_list = json.load(f)
    
    with open(out_file, "w+", encoding="utf8") as writer:
        for in_obj in input_list:
            out_dict = in_obj
            wav_name = out_dict["audio_id"].split("/")[-1]
            out_dict["audio"] = os.path.join(wav_dir, wav_name)

            new_line = json.dumps(out_dict, ensure_ascii=False)
            writer.write(new_line + "\n")


handle_mmau()
