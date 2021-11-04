# FullSubNet+

This Git repository for the official PyTorch implementation of ["FullSubNet+: Channel Attention FullSubNet with Complex Spectrograms for Speech Enhancement"](),  submitted to ICASSP 2021.

▶[[Demo](https://hit-thusz-rookiecj.github.io/fullsubnet-plus.github.io/)]



## Requirements

\- Linux or macOS 

\- python>=3.6

\- Anaconda or Miniconda

\- NVIDIA GPU + CUDA CuDNN (CPU is **not** be supported)

 <br/><br/>

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

<br/><br/><br/><br/> 

## Start Up

### Clone

```shell
git https://github.com/hit-thusz-RookieCJ/FullSubNet-plus.git
cd FullSubNet-plus
```

<br/><br/>

### Data preparation

#### Train data

Please prepare your data in the data dir as like:

- data/DNS-Challenge/DNS-Challenge-interspeech2020-master/
- data/DNS-Challenge/DNS-Challenge-master/

and set the train dir in the script `run.sh`.

then:

```shell
source activate speech_enhance
bash run.sh 0   # peprare training list or meta file
```

#### Test(eval) data

Please prepare your test cases dir like: `data/test_cases_<name>`, and set the test dir in the script `run.sh`.

<br/><br/>

### Training

First, you need to modify the various configurations in `config/train.toml` for training.

Then you can run training:

```shell
source activate speech_enhance
bash run.sh 1   
```

<br/><br/>

### Inference

After training, you can enhance noisy speech.  Before inference, you first need to modify the configuration in `config/inference.toml`.

Then you can run inference:

```shell
source activate speech_enhance
bash run.sh 2
```

Or you can just use `inference.sh`:

```shell
source activate speech_enhance
bash inference.sh
```

<br/><br/>

### Eval

Calculating metrics (SI_SDR, STOI, WB_PESQ, NB_PESQ, etc.) :

```shell
bash metrics.sh
```

