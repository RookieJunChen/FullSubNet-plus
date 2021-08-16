import os
from glob import glob
import argparse

def gen_lst(args):
    wav_lst = glob(os.path.join(args.dataset_dir, "**/*.wav"), recursive=True)
    os.makedirs(os.path.dirname(args.output_lst), exist_ok=True)
    fc = open(args.output_lst, "w")
    for one_wav in wav_lst:
        fc.write(f"{one_wav}\n")
    fc.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--output_lst", type=str, default="")
    args = parser.parse_args()

    gen_lst(args)
