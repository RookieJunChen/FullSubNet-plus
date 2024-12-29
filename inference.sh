#!/usr/bin/env bash



# do enhance(denoise)
CUDA_VISIBLE_DEVICES='0' /usr/bin/python3 -m speech_enhance.tools.inference \
  -C config/inference.toml \
  -M /workspace/project-nas-11025-sh/speech_enhance/egs/DNS-master/s1_16k/logs/Complex_amp_input_allattention_DeepTCNFCFullSubNet/train_amp_attention_complex_fullsubnet/checkpoints/best_model.tar \
  -I /workspace/project-nas-11025-sh/speech_enhance/data/DNS-Challenge/DNS-Challenge-interspeech2020-master/datasets/test_set/synthetic/with_reverb/noisy \
  -O /workspace/project-nas-11025-sh/speech_enhance/case/with_reverb/fullsubnet+


# Normalized to -6dB (optional)
sdir="/workspace/project-nas-11025-sh/speech_enhance/case/with_reverb/fullsubnet/enhanced_0073"
fdir="/workspace/project-nas-11025-sh/speech_enhance/case/with_reverb/fullsubnet/fullsubnet_norm"

softfiles=$(find $sdir -name "*.wav")
for file in ${softfiles}
do 
  length=${#sdir}+1
  file=${file:$length}
  f=$sdir/$file
  echo $f
  dstfile=$fdir/$file
  echo $dstfile
  sox $f -b16 $dstfile rate -v -b 99.7 16k norm -6
done


