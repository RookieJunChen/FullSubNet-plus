import argparse
import os
import sys

import toml

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from audio_zen.utils import initialize_module


def main(config, checkpoint_path, output_dir):
    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir
    )
    inferencer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("-C", "--configuration", type=str, required=True, help="Config file.")
    parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="The path of the model's checkpoint.")
    parser.add_argument('-I', '--dataset_dir_list', help='delimited list input',
                        type=lambda s: [item.strip() for item in s.split(',')])
    parser.add_argument("-O", "--output_dir", type=str, required=True, help="The path for saving enhanced speeches.")
    args = parser.parse_args()

    configuration = toml.load(args.configuration)
    checkpoint_path = args.model_checkpoint_path
    output_dir = args.output_dir
    if len(args.dataset_dir_list) > 0:
        print(f"use specified dataset_dir_list: {args.dataset_dir_list}, instead of in config")
        configuration["dataset"]["args"]["dataset_dir_list"] = args.dataset_dir_list

    main(configuration, checkpoint_path, output_dir)
