
# FullSubNet+

This Git repository for the official PyTorch implementation of **"[FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement](https://arxiv.org/abs/2203.12188)"**,  accepted by ICASSP 2022.

📜[[Full Paper](https://arxiv.org/abs/2203.12188)] ▶[[Demo](https://hit-thusz-rookiecj.github.io/fullsubnet-plus.github.io/)] 💿[[Checkpoint](https://drive.google.com/file/d/1UJSt1G0P_aXry-u79LLU_l9tCnNa2u7C/view)]



## Requirements

- Linux or macOS 

- python>=3.6

- Anaconda or Miniconda

- NVIDIA GPU + CUDA CuDNN (CPU can also be supported)



### Environment && Installation

Install Anaconda or Miniconda, and then install conda and pip packages:

```shell
# Create conda environment
conda create --name speech_enhance python=3.6
conda activate speech_enhance

# Install conda packages
# Check python=3.8, cudatoolkit=10.2, pytorch=1.7.1, torchaudio=0.7
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install tensorboard joblib matplotlib

# Install pip packages
# Check librosa=0.8
pip install Cython
pip install librosa pesq pypesq pystoi tqdm toml colorful mir_eval torch_complex

# (Optional) If you want to load "mp3" format audio in your dataset
conda install -c conda-forge ffmpeg
```



### Quick Usage

Clone the repository:

```shell
git clone https://github.com/hit-thusz-RookieCJ/FullSubNet-plus.git
cd FullSubNet-plus
```

Download the [pre-trained checkpoint](https://drive.google.com/file/d/1UJSt1G0P_aXry-u79LLU_l9tCnNa2u7C/view), and input commands:

```shell
source activate speech_enhance
python -m speech_enhance.tools.inference \
  -C config/inference.toml \
  -M $MODEL_DIR \
  -I $INPUT_DIR \
  -O $OUTPUT_DIR
```

<br/> 

## Start Up

### Clone

```shell
git clone https://github.com/hit-thusz-RookieCJ/FullSubNet-plus.git
cd FullSubNet-plus
```



### Data preparation

#### Train data

Please prepare your data in the data dir as like:

- data/DNS-Challenge/DNS-Challenge-interspeech2020-master/
- data/DNS-Challenge/DNS-Challenge-master/

and set the train dir in the script `run.sh`.

Then:

```shell
source activate speech_enhance
bash run.sh 0   # peprare training list or meta file
```

#### Test data

Please prepare your test cases dir like: `data/test_cases_<name>`, and set the test dir in the script `run.sh`.



### Training

First, you need to modify the various configurations in `config/train.toml` for training.

Then you can run training:

```shell
source activate speech_enhance
bash run.sh 1   
```



### Inference

After training, you can enhance noisy speech.  Before inference, you first need to modify the configuration in `config/inference.toml`.

You can also run inference:

```shell
source activate speech_enhance
bash run.sh 2
```

Or you can just use `inference.sh`:

```shell
source activate speech_enhance
bash inference.sh
```





### Eval

Calculating bjective metrics (SI_SDR, STOI, WB_PESQ, NB_PESQ, etc.) :

```shell
bash metrics.sh
```

Obtain subjective scores (DNS_MOS):
```shell
python ./speech_enhance/tools/dns_mos.py --testset_dir $YOUR_TESTSET_DIR --score_file $YOUR_SAVE_DIR
```



## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{chen2022fullsubnet+,
  title={FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement},
  author={Chen, Jun and Wang, Zilin and Tuo, Deyi and Wu, Zhiyong and Kang, Shiyin and Meng, Helen},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7857--7861},
  year={2022},
  organization={IEEE}
}
```