#!/usr/bin/env bash

/usr/bin/python3 speech_enhance/tools/calculate_metrics.py \
  -R /workspace/project-nas-11025-sh/speech_enhance/data/DNS-Challenge/DNS-Challenge-interspeech2020-master/datasets/test_set/synthetic/with_reverb/clean \
  -E /workspace/project-nas-11025-sh/speech_enhance/case/with_reverb/fullsubnet+/enhanced_0194 \
  -M SI_SDR,STOI,WB_PESQ,NB_PESQ \
  -S DNS_1 \
  -D /workspace/project-nas-11025-sh/speech_enhance/egs/DNS-master/s1_16k/mertrics/with_reverb_fullsubnet+/