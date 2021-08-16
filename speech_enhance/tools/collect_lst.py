import os
import random
import sys
from pathlib import Path
import librosa
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from audio_zen.acoustics.mask import is_clipped, load_wav, activity_detector


def offset_and_limit(data_list, offset, limit):
    data_list = data_list[offset:]
    if limit:
        data_list = data_list[:limit]
    return data_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FullSubNet")
    parser.add_argument('-candidate_datasets', '--candidate_datasets', help='delimited list input',
                        type=lambda s: [item for item in s.split(',')])
    parser.add_argument("-dist_file", "--dist_file", required=True, type=str, help="output lst")
    parser.add_argument("-sr", "--sr", type=int, default=16000, help="sample rate")
    parser.add_argument("-wav_min_second", "--wav_min_second", type=int, default=3, help="the min length of a wav")
    parser.add_argument("-activity_threshold", "--activity_threshold", type=float, default=0.6,
                        help="the activity threshold of speech/sil")
    parser.add_argument("-total_hrs", "--total_hrs", type=int, default=30, help="the length in time of wav(s)")

    args = parser.parse_args()

    candidate_datasets = args.candidate_datasets
    dataset_limit = None
    dataset_offset = 0
    dist_file = args.dist_file

    # 声学参数
    sr = args.sr
    wav_min_second = args.wav_min_second
    activity_threshold = args.activity_threshold
    total_hrs = args.total_hrs  # 计划收集语音的总时长

    all_wav_path_list = []
    output_wav_path_list = []
    accumulated_time = 0.0

    is_clipped_wav_list = []
    is_low_activity_list = []
    is_too_short_list = []

    for dataset_path in candidate_datasets:
        dataset_path = Path(dataset_path).expanduser().absolute()
        all_wav_path_list += librosa.util.find_files(dataset_path.as_posix(), ext=["wav"])

    all_wav_path_list = offset_and_limit(all_wav_path_list, dataset_offset, dataset_limit)
    random.shuffle(all_wav_path_list)

    for wav_file_path in tqdm(all_wav_path_list, desc="Checking"):
        y = load_wav(wav_file_path, sr=sr)
        wav_duration = len(y) / sr
        wav_file_user_path = wav_file_path.replace(Path(wav_file_path).home().as_posix(), "~")

        is_clipped_wav = is_clipped(y)
        is_low_activity = activity_detector(y) < activity_threshold
        is_too_short = wav_duration < wav_min_second

        if is_too_short:
            is_too_short_list.append(wav_file_user_path)
            continue

        if is_clipped_wav:
            is_clipped_wav_list.append(wav_file_user_path)
            continue

        if is_low_activity:
            is_low_activity_list.append(wav_file_user_path)
            continue

        if (not is_clipped_wav) and (not is_low_activity) and (not is_too_short):
            accumulated_time += wav_duration
            output_wav_path_list.append(wav_file_user_path)

        if accumulated_time >= (total_hrs * 3600):
            break

    os.makedirs(os.path.dirname(dist_file.as_posix()), exist_ok=True)
    with open(dist_file.as_posix(), 'w') as f:
        f.writelines(f"{file_path}\n" for file_path in output_wav_path_list)

    print("=" * 70)
    print("Speech Preprocessing")
    print(f"\t Original files: {len(all_wav_path_list)}")
    print(f"\t Selected files: {accumulated_time / 3600} hrs, {len(output_wav_path_list)} files.")
    print(f"\t is_clipped_wav: {len(is_clipped_wav_list)}")
    print(f"\t is_low_activity: {len(is_low_activity_list)}")
    print(f"\t is_too_short: {len(is_too_short_list)}")
    print(f"\t dist file:")
    print(f"\t {dist_file.as_posix()}")
    print("=" * 70)
