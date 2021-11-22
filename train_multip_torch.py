# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data_1 import TextMelMultispeakerDataset, TextMelMultispeakerBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

import os ###
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import glob

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)

#import pdb
#pdb.set_trace()
#nsymbols = 37 + 1 if add_blank else 37
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale

n_speakers = params.n_speakers
gin_channels_spk = params.gin_channels_spk

n_emotions = params.n_emotions
gin_channels_emotion = params.gin_channels_emotion

def run_train(rank, n_gpus):  ###
    if rank == 0:  ###
        print('Initializing logger...')
        logger = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    print(n_gpus)
    print(rank)
    
    dist.init_process_group(backend='nccl', init_method='env://',  ###
                            world_size=n_gpus, rank=rank)
    torch.cuda.set_device(rank)  ###

    print('Initializing data loaders...')
    train_dataset = TextMelMultispeakerDataset(train_filelist_path, cmudict_path, add_blank)
    train_sampler = torch.utils.data.distributed.DistributedSampler(  ###
      train_dataset,
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)

    batch_collate = TextMelMultispeakerBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False, sampler=train_sampler)
    # loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        #collate_fn=batch_collate, drop_last=True,
                        #num_workers=4, shuffle=False)

    if rank == 0:
        test_dataset = TextMelMultispeakerDataset(valid_filelist_path, cmudict_path, add_blank)
        print('Logging test batch...')
        test_batch = test_dataset.sample_test_batch(size=params.test_size)
        for i, item in enumerate(test_batch):
            mel = item['y']
            logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
            save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')


    print('Initializing model...')
    model = GradTTS(nsymbols, n_enc_channels, filter_channels, filter_channels_dp,
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                    n_feats, dec_dim, beta_min, beta_max, pe_scale, n_speakers, n_emotions, gin_channels_spk, gin_channels_emotion).cuda(rank)
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    model = torch.nn.DataParallel(model)  ###

    

    ###
    print('Load from checkpoint...')
    epoch_str = 1
    try:
        _, _, _, epoch_str = load_checkpoint(latest_checkpoint_path(log_dir, "G_*.pth"), model, optimizer)
        epoch_str += 1
        optimizer.step_num = (epoch_str - 1) * len(loader)
    except:
        print("No checkpoints...")


    print('Start training...')
    iteration = 0
    for epoch in range(epoch_str, n_epochs + 1):
        if rank == 0:
            loader.sampler.set_epoch(epoch)
            model.train()
            dur_losses = []
            prior_losses = []
            diff_losses = []
            with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
                for batch_idx, batch in enumerate(progress_bar):
                    model.zero_grad()
                    x, x_lengths = batch['x'].cuda(rank, non_blocking=True), batch['x_lengths'].cuda(rank, non_blocking=True)
                    y, y_lengths = batch['y'].cuda(rank, non_blocking=True), batch['y_lengths'].cuda(rank, non_blocking=True)
                    g1 = batch['sid']
                    g2 = batch['eid'].cuda(rank)
                    dur_loss, prior_loss, diff_loss = model.module.compute_loss(x, x_lengths,
                                                                 y, y_lengths,
                                                                 g1=g1, g2=g2,
                                                                 out_size=out_size)
                    loss = sum([dur_loss, prior_loss, diff_loss])
                    loss.backward()

                    enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.encoder.parameters(),
                                                           max_norm=1)
                    dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.module.decoder.parameters(),
                                                           max_norm=1)
                    optimizer.step()

                    if rank == 0:
                        logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                        logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                        logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                        logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                        logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                        dur_losses.append(dur_loss.item())
                        prior_losses.append(prior_loss.item())
                        diff_losses.append(diff_loss.item())

                        if batch_idx % 5 == 0:
                            msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                            progress_bar.set_description(msg)

                        iteration += 1

            if rank == 0:
                log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
                log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
                log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
                with open(f'{log_dir}/train.log', 'a') as f:
                    f.write(log_msg)
            if epoch % params.save_every > 0:
                continue
            if rank == 0:
                model.eval()
                with torch.no_grad():
                    for i, item in enumerate(test_batch):
                        x = item['x'].to(torch.long).unsqueeze(0).cuda(rank, non_blocking=True)
                        x_lengths = torch.LongTensor([x.shape[-1]]).cuda(rank, non_blocking=True)
                        g1 = item['sid'].unsqueeze(0).cuda(rank, non_blocking=True)
                        g2 = item['eid'].unsqueeze(0).cuda(rank, non_blocking=True)

                        #print(g.shape, 'sid')
                        y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50, g1=g1, g2=g2)
                        logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                        logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                        logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                        save_plot(y_enc.squeeze().cpu(),
                          f'{log_dir}/generated_enc_{i}.png')
                        save_plot(y_dec.squeeze().cpu(),
                          f'{log_dir}/generated_dec_{i}.png')
                        save_plot(attn.squeeze().cpu(),
                          f'{log_dir}/alignment_{i}.png')
            save_checkpoint(model, optimizer, learning_rate, epoch, os.path.join(log_dir, "G_{}.pth".format(epoch)))
    
def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = 1
    if 'iteration' in checkpoint_dict.keys():
        iteration = checkpoint_dict['iteration']
    if 'learning_rate' in checkpoint_dict.keys():
        learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    print("Loaded checkpoint '{}' (iteration {})" .format(
                      checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    print("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8000'
    n_gpus = torch.cuda.device_count()
    mp.spawn(run_train, nprocs=n_gpus, args=(n_gpus,))
