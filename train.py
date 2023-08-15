# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main entry point for training StyleGAN and ProGAN networks."""


import copy
import dnnlib
from dnnlib import EasyDict

import config
from metrics import metric_base

train         = EasyDict(run_func_name='training.training_loop.training_loop')         # Options for training loop.
G             = EasyDict(func_name='training.networks_stylegan.G_style')               # Options for generator network.
D             = EasyDict(func_name='training.networks_stylegan.D_basic')               # Options for discriminator network.
G_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for generator optimizer.
D_opt         = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                          # Options for discriminator optimizer.
G_loss        = EasyDict(func_name='training.loss.G_logistic_nonsaturating')           # Options for generator loss.
D_loss        = EasyDict(func_name='training.loss.D_logistic_simplegp', r1_gamma=10.0) # Options for discriminator loss.
dataset       = EasyDict()                                                             # Options for load_dataset().
sched         = EasyDict()                                                             # Options for TrainingSchedule.
grid          = EasyDict(size='4k', layout='random')                                   # Options for setup_snapshot_image_grid().
metrics       = [metric_base.fid50k]                                                   # Options for MetricGroup.
submit_config = dnnlib.SubmitConfig()                                                  # Options for dnnlib.submit_run().
tf_config     = {'rnd.np_random_seed': 1000}                                           # Options for tflib.init_tf().

desc = 'sgan' + '-ffhq'
dataset = EasyDict(tfrecord_dir='ffhq')
train.mirror_augment = True
#desc += '-ffhq512';  dataset = EasyDict(tfrecord_dir='ffhq', resolution=512); train.mirror_augment = True
#desc += '-ffhq256';  dataset = EasyDict(tfrecord_dir='ffhq', resolution=256); train.mirror_augment = True
#desc += '-celebahq'; dataset = EasyDict(tfrecord_dir='celebahq');             train.mirror_augment = True
#desc += '-bedroom';  dataset = EasyDict(tfrecord_dir='lsun-bedroom-full');    train.mirror_augment = False
#desc += '-car';      dataset = EasyDict(tfrecord_dir='lsun-car-512x384');     train.mirror_augment = False
#desc += '-cat';      dataset = EasyDict(tfrecord_dir='lsun-cat-full');        train.mirror_augment = False

# Number of GPUs.
#desc += '-1gpu'; submit_config.num_gpus = 1; sched.minibatch_base = 4; sched.minibatch_dict = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4}
#desc += '-2gpu'; submit_config.num_gpus = 2; sched.minibatch_base = 8; sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}
#desc += '-4gpu'; submit_config.num_gpus = 4; sched.minibatch_base = 16; sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
desc += '-8gpu'
submit_config.num_gpus = 8
sched.minibatch_base = 32
sched.minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}

# Default options.
train.total_kimg = 25000
sched.lod_initial_resolution = 8
sched.G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
sched.D_lrate_dict = EasyDict(sched.G_lrate_dict)
    #desc += '-fp16'; G.dtype = 'float16'; D.dtype = 'float16'; G.pixelnorm_epsilon=1e-4; G_opt.use_loss_scaling = True; D_opt.use_loss_scaling = True; sched.max_minibatch_per_gpu = {512: 16, 1024: 8}

    # Disable individual features.
    #desc += '-nogrowing'; sched.lod_initial_resolution = 1024; sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 10000
    #desc += '-nopixelnorm'; G.use_pixelnorm = False
    #desc += '-nowscale'; G.use_wscale = False; D.use_wscale = False
    #desc += '-noleakyrelu'; G.use_leakyrelu = False
    #desc += '-nosmoothing'; train.G_smoothing_kimg = 0.0
    #desc += '-norepeat'; train.minibatch_repeats = 1
    #desc += '-noreset'; train.reset_opt_for_new_lod = False

    # Special modes.
    #desc += '-BENCHMARK'; sched.lod_initial_resolution = 4; sched.lod_training_kimg = 3; sched.lod_transition_kimg = 3; train.total_kimg = (8*2+1)*3; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1000; train.network_snapshot_ticks = 1000
    #desc += '-BENCHMARK0'; sched.lod_initial_resolution = 1024; train.total_kimg = 10; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1000; train.network_snapshot_ticks = 1000
    #desc += '-VERBOSE'; sched.tick_kimg_base = 1; sched.tick_kimg_dict = {}; train.image_snapshot_ticks = 1; train.network_snapshot_ticks = 100
    #desc += '-GRAPH'; train.save_tf_graph = True
    #desc += '-HIST'; train.save_weight_histograms = True

#----------------------------------------------------------------------------
# Main entry point for training.
# Calls the function indicated by 'train' using the selected options.

def main():
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config)
    kwargs.submit_config = copy.deepcopy(submit_config)
    kwargs.submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    kwargs.submit_config.run_dir_ignore += config.run_dir_ignore
    kwargs.submit_config.run_desc = desc
    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
