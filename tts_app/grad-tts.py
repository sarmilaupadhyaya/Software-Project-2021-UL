# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 15:10:21 2021

@author: rasul
"""
import argparse
import json
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm
from scipy.io.wavfile import write

import torch

# For Grad-TTS
import sys
sys.path.append('Grad-TTS/')
import params
from model import GradTTS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

# For HiFi-GAN
sys.path.append('Grad-TTS/hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

N_SPKS = 247  # 247 for Libri-TTS checkpoint and 1 for LJSpeech single speaker checkpoint

# Note: besides Libri-TTS checkpoint we open-source 2 LJSpeech checkpoints of Grad-TTS.
# These 2 are the same models but trained with different positional encoding scale:
#   * x1 ("grad-tts-old.pt", ICML 2021 sumbission model)
#   * x1000 ("grad-tts.pt")
# To use the former set Grad-TTS argument pe_scale=1 and to use the latter set pe_scale=1000.
# To use Libri-TTS checkpoint use pe_scale=1000.

generator = GradTTS(len(symbols)+1, N_SPKS, params.spk_emb_dim, params.n_enc_channels, params.filter_channels,
                    params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                    params.enc_kernel, params.enc_dropout, params.window_size,
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max,
                    pe_scale=1000)  # pe_scale=1 for `grad-tts-old.pt`
generator.load_state_dict(torch.load('./Grad-TTS/checkpts/grad-tts-libri-tts.pt', map_location=lambda loc, storage: loc))
_ = generator.cuda().eval()
print(f'Number of encoder parameters: {generator.encoder.nparams}')
print(f'Number of decoder parameters: {generator.decoder.nparams}')
print(f'Number of total parameters: {generator.nparams}')

cmu = cmudict.CMUDict('./Grad-TTS/resources/cmu_dictionary')

with open('./Grad-TTS/checkpts/hifigan-config.json') as f:
    h = AttrDict(json.load(f))
hifigan = HiFiGAN(h)
hifigan.load_state_dict(torch.load('./Grad-TTS/checkpts/hifigan.pt',
                                   map_location=lambda loc, storage: loc)['generator'])
_ = hifigan.cuda().eval()
hifigan.remove_weight_norm()

text = "Here are the match lineups for the Colombia Haiti match."
SPEAKER_ID = 15  # set speaker id if you are using multi-speaker model, ignore otherwise
x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

t = dt.datetime.now()
y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=10, temperature=1.5,
                                       stoc=False, spk=torch.LongTensor([SPEAKER_ID]).cuda() if N_SPKS > 1 else None,
                                       length_scale=0.91)
t = (dt.datetime.now() - t).total_seconds()