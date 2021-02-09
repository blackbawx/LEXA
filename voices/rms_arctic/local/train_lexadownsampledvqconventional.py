"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --conf=<json>             Path of configuration file (json).
    --gpu-id=<N>               ID of the GPU to use [default: 0]
    --exp-dir=<dir>           Experiment directory
    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    --hparams=<parmas>        Hyper parameters [default: ].
    --log-event-path=<dir>    Log Path [default: exp/log_tacotronOne]
    -h, --help                Show this help message and exit
"""
import os, sys
from docopt import docopt
args = docopt(__doc__)
print("Command line args:\n", args)
gpu_id = args['--gpu-id']
print("Using GPU ", gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id


from collections import defaultdict

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)
##############################################
from utils.misc import *
from utils import audio
from utils.plot import plot_alignment
from tqdm import tqdm, trange
from util import *
from model import LexatronDownsampledVQConventional as Tacotron
from judith.experiment_tracking import RemoteTracker


import json

import torch
from torch.utils import data as data_utils
from torch.autograd import Variable
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np

from os.path import join, expanduser

import tensorboard_logger
from tensorboard_logger import *
from hyperparameters import hyperparameters

vox_dir ='vox'

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
use_multigpu = None

hparams = hyperparameters()
print(hparams)
fs = hparams.sample_rate


use_assistant = 1
normalize_latents = True


def train(model, train_loader, val_loader, optimizer,
          init_lr=0.002,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None,
          clip_thresh=1.0, assistant=None):
    model.train()
    if use_cuda:
        model = model.cuda()
    linear_dim = model.linear_dim

    criterion = nn.L1Loss()
    running_entropy = 0.

    global global_step, global_epoch
    while global_epoch < nepochs:
        h = open(logfile_name, 'a')
        running_loss = 0.

        for step, (input_lengths, mel, y) in tqdm(enumerate(train_loader)):

            # Decay learning rate
            #current_lr = learning_rate_decay(init_lr, global_step)
            #for param_group in optimizer.param_groups:
            #    param_group['lr'] = current_lr

            optimizer.zero_grad()

            # Sort by length
            sorted_lengths, indices = torch.sort(
                input_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            mel, y = mel[indices], y[indices]

            # Feed data
            mel, y = Variable(mel), Variable(y)
            if use_cuda:
                mel, y = mel.cuda(), y.cuda()

            # Multi GPU Configuration
            if use_multigpu:
               outputs,  r_, o_ = data_parallel_workaround(model, (x, mel))
               mel_outputs, linear_outputs, attn = outputs[0], outputs[1], outputs[2]
 
            else:
                mel_outputs, linear_outputs, attn, entropy, classes, vq_penalty, encoder_penalty = model(mel)

            # Loss
            mel_loss = criterion(mel_outputs, mel)
            n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
            linear_loss = 0.5 * criterion(linear_outputs, y) \
                + 0.5 * criterion(linear_outputs[:, :, :n_priority_freq],
                                  y[:, :, :n_priority_freq])
            encoder_weight = 0.25 * min(1, max(0.1, global_step / 1000 - 1)) # https://github.com/mkotha/WaveRNN/blob/74b839b57a7e128b3f8f0b4eb224156c1e5e175d/models/vqvae.py#L209

            loss = mel_loss + linear_loss + vq_penalty + encoder_penalty * encoder_weight 

            if global_step > 0 and global_step % hparams.save_states_interval == 0:
                save_states(
                    global_step, mel_outputs, linear_outputs, attn, y,
                    None, checkpoint_dir)
                #visualize_latent_embedding0(model, checkpoint_dir, global_step)

            if global_step > 0 and global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)

            # Update
            loss.backward(retain_graph=False)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                 model.parameters(), clip_thresh)
            optimizer.step()
            if normalize_latents:
               model.vq.after_update()

            # Logs
            log_value("loss", float(loss.item()), global_step)
            log_value("mel loss", float(mel_loss.item()), global_step)
            log_value("linear loss", float(linear_loss.item()), global_step)
            log_value("gradient norm", grad_norm, global_step)
            #log_value("learning rate", current_lr, global_step)
            log_histogram("Last Linear Weights", model.last_linear.weight.detach().cpu(), global_step)
            global_step += 1
            running_entropy += entropy
            running_loss += loss.item()
            classes = ' '.join(str(k) for k in classes.cpu().detach().numpy().tolist())
            #print("Latents: ", classes)

            if use_assistant and assistant is not None:
              assistant.log_scalar("Encoder Penalty", encoder_penalty)
              assistant.log_scalar("Encoder Weight", encoder_weight)
              assistant.log_scalar("VQ Penalty", vq_penalty)
              assistant.log_scalar("loss", float(loss.item()))
              assistant.log_scalar("Mel Loss", float(mel_loss.item()))
              assistant.log_scalar("Linear Loss", float(linear_loss.item()))
              assistant.log_scalar("Gradient Norm", float(grad_norm))
              assistant.log_scalar("Average Entropy", float(running_entropy/global_step))
              assistant.log_scalar("Entropy", float(entropy))
              assistant.log_text('latents', classes)

              #assistant.log_scalar("Learning Rate", float(current_lr))

        averaged_loss = running_loss / (len(train_loader))
        log_value("loss (per epoch)", averaged_loss, global_epoch)
        h.write("Loss after epoch " + str(global_epoch) + ': '  + format(running_loss / (len(train_loader))) + '\n')
        h.close() 
        #sys.exit()

        global_epoch += 1


if __name__ == "__main__":

    exp_dir = args["--exp-dir"]
    checkpoint_dir = args["--exp-dir"] + '/checkpoints'
    checkpoint_path = args["--checkpoint-path"]
    log_path = args["--exp-dir"] + '/tracking'
    conf = args["--conf"]
    #hparams.parse(args["--hparams"])

    # Override hyper parameters
    if conf is not None:
        with open(conf) as f:
            hparams.update_params(f)
    print(hparams)
    print(hparams.batch_size)
    #sys.exit()

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    logfile_name = log_path + '/logfile'
    h = open(logfile_name, 'w')
    h.close()

    # Vocab size
    with open(vox_dir + '/' + 'etc/ids_phones.json') as  f:
       ph_ids = json.load(f)

    ph_ids = dict(ph_ids)
    print(ph_ids)

    idsdict_file = checkpoint_dir + '/ids_phones.json'

    with open(idsdict_file, 'w') as outfile:
       json.dump(ph_ids, outfile)



    feats_name = 'phones'
    X_train = categorical_datasource( vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' +  'festival/falcon_' + feats_name, ph_ids)
    X_val = CategoricalDataSource(vox_dir + '/' +  'fnames.val', vox_dir + '/' +  'etc/falcon_feats.desc', feats_name,  feats_name, ph_ids)

    feats_name = 'lspec'
    Y_train = float_datasource(vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    Y_val = FloatDataSource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    feats_name = 'mspec'
    Mel_train = float_datasource(vox_dir + '/' + 'fnames.train', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)
    Mel_val = FloatDataSource(vox_dir + '/' + 'fnames.val', vox_dir + '/' + 'etc/falcon_feats.desc', feats_name, vox_dir + '/' + 'festival/falcon_' + feats_name)

    # Dataset and Dataloader setup
    trainset = PyTorchDataset(X_train, Mel_train, Y_train)
    train_loader = data_utils.DataLoader(
        trainset, batch_size=int(hparams.batch_size),
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_lexa, pin_memory=hparams.pin_memory)

    valset = PyTorchDataset(X_val, Mel_val, Y_val)
    val_loader = data_utils.DataLoader(
        valset, batch_size=int(hparams.batch_size),
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn_lexa, pin_memory=hparams.pin_memory)

    # Model
    model = Tacotron(n_vocab=1+len(ph_ids),
                     embedding_dim=256,
                     num_latentclasses=64,
                     mel_dim=hparams.num_mels,
                     linear_dim=hparams.num_freq,
                     r=hparams.outputs_per_step,
                     padding_idx=hparams.padding_idx,
                     use_memory_mask=hparams.use_memory_mask,
                     normalize_latents=normalize_latents
                     )
    model = model.cuda()
    #model = DataParallelFix(model)

    optimizer = optim.Adam(model.parameters(),
                           lr=hparams.initial_learning_rate, betas=(
                               hparams.adam_beta1, hparams.adam_beta2),
                           weight_decay=hparams.weight_decay)

    # Load checkpoint
    if checkpoint_path:
        print("Load checkpoint from: {}".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            global_step = int(checkpoint["global_step"])
            global_epoch = int(checkpoint["global_epoch"])
        except:
            # TODO
            pass

    # Setup tensorboard logger
    tensorboard_logger.configure(log_path)

    #print(hparams_debug_string())

    # Train!
    try:
        assistant = None
        exp_name = os.path.basename(exp_dir)
        if use_assistant:
           assistant = RemoteTracker(exp_name, projects_dir='LEXA')
        train(model, train_loader, val_loader, optimizer,
              init_lr=hparams.initial_learning_rate,
              checkpoint_dir=checkpoint_dir,
              checkpoint_interval=hparams.checkpoint_interval,
              nepochs=int(hparams.nepochs),
              clip_thresh=hparams.clip_thresh,
              assistant=assistant)
    except KeyboardInterrupt:
        save_checkpoint(
            model, optimizer, global_step, checkpoint_dir, global_epoch)

    print("Finished")
    sys.exit(0)

