import os
import shutil
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from .logger import log


def touch_dir(d):
    os.makedirs(d, exist_ok=True)


def is_file_exists(f):
    return os.path.exists(f)


def check_file_exists(f):
    if not os.path.exists(f):
        log(f"not found file: {f}")
        assert False, f"not found file: {f}"


def read_lines(data_path):
    lines = []
    with open(data_path, encoding="utf-8") as fr:
        for line in fr.readlines():
            if len(line.strip().replace(" ", "")):
                lines.append(line.strip())
    # log("read {} lines from {}".format(len(lines), data_path))
    # log("example(last) {}\n".format(lines[-1]))
    return lines


def write_lines(data_path, lines):
    with open(data_path, "w", encoding="utf-8") as fw:
        for line in lines:
            fw.write("{}\n".format(line))
    # log("write {} lines to {}".format(len(lines), data_path))
    # log("example(last line): {}\n".format(lines[-1]))
    return


def get_name_from_path(abs_path):
    return ".".join(os.path.basename(abs_path).split(".")[:-1])


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        return


def load_hparams(yaml_path):
    with open(yaml_path, encoding="utf-8") as yaml_file:
        hparams = yaml.safe_load(yaml_file)
    return AttrDict(hparams)


def dump_hparams(yaml_path, hparams):
    touch_dir(os.path.dirname(yaml_path))
    with open(yaml_path, "w") as fw:
        yaml.dump(hparams, fw)
    log("save hparams to {}".format(yaml_path))
    return


def get_all_wav_path(file_dir):
    wav_list = []
    for path, dir_list, file_list in os.walk(file_dir):
        for file_name in file_list:
            if file_name.endswith(".wav") or file_name.endswith(".WAV"):
                wav_path = os.path.join(path, file_name)
                wav_list.append(wav_path)
    return sorted(wav_list)


def clean_and_new_dir(data_dir):
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)
    return


def generate_dir_tree(synth_dir, dir_name_list, del_old=False):
    os.makedirs(synth_dir, exist_ok=True)
    dir_path_list = []
    if del_old:
        shutil.rmtree(synth_dir, ignore_errors=True)
    for name in dir_name_list:
        dir_path = os.path.join(synth_dir, name)
        dir_path_list.append(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    return dir_path_list


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


if __name__ == '__main__':
    pass
