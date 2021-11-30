# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:26:48 2021

@author: rasul
"""
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

def load_grad_tts(checkpoint):
    print('Initializing Grad-TTS...')
    generator = GradTTS(len(symbols)+1, params.n_spks, params.spk_emb_dim,
                        params.n_enc_channels, params.filter_channels,
                        params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                        params.enc_kernel, params.enc_dropout, params.window_size,
                        params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)
    generator.load_state_dict(torch.load(checkpoint, map_location=lambda loc, storage: loc))
    _ = generator.eval()
    print(f'Number of parameters: {generator.nparams}')
    return generator

def load_hifi():
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder
    
def main(text, checkpt, timesteps=10, speaker_id=None):
    if not isinstance(speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([speaker_id]).cuda()
    else:
        spk = None
    generator = load_grad_tts(checkpt)
    vocoder = load_hifi()
    texts = [line.strip() for line in text.split("\n") if not line.isspace()]
    cmu = cmudict.CMUDict('./resources/cmu_dictionary')
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            print(f'Synthesizing {i} text...', end=' ')
            x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols)))[None]
            x_lengths = torch.LongTensor([x.shape[-1]])
            
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=timesteps, temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=0.91)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
            
            write(f'./out/sample_{i}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')    
if __name__ == '__main__':
    main()

