import os
from glob import glob
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

def resample_one_wav(wav_file, output_wav_file):
    if not os.path.exists(wav_file):
        print(f"not found {wav_file}, return")
        return
    ####
    #print(wav_file)
    #print(output_wav_file)
    os.makedirs(os.path.dirname(output_wav_file), exist_ok=True)
    cmd = f"sox {wav_file} -b16 {output_wav_file} rate -v -b 99.7 16k"
    os.system(cmd)

def resample_dir(args):
    ### get all wavs
    wav_lst = glob(os.path.join(args.dataset_dir, "**/*.wav"), recursive=True)
    os.makedirs(args.output_dir, exist_ok=True)
    ###
    num_workers = 40
    Parallel(n_jobs=num_workers)(
        delayed(resample_one_wav)(wav_file, wav_file.replace(args.dataset_dir, args.output_dir)) for wav_file in wav_lst)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    resample_dir(args)
