

# Core Imports
import time
import argparse
import random

# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np

# Internal Project Imports
from pretrained.vgg import Vgg16Pretrained
from utils import misc as misc
from utils.misc import load_path_for_pytorch
from utils.stylize import produce_stylization

# These should come from our helpers
################################################################################

from itertools import cycle, islice

def cycle_args(function, *args):
     """calls a function multiple times with each arg cycled to max count,  (func, arg_a_list, arg_b_list...)"""
     max_len = max(map(len,args))
     args = list(args)
     for i, arg in enumerate(args):
          args[i] = list(islice(cycle(arg), max_len))
          
     print('Running', function.__name__, max_len, 'times')

     for i in range(max_len):
          function( *[arg[i] for arg in args] ) 

import os
def filename(path):
    '''returns just filename without extension from full path'''
    return os.path.splitext(os.path.split(path)[-1])[0]

################################################################################


# style transfer with slightly simplifed signiture, for easy vvvv experiments
def style_transfer(content_path, style_path, output_path, style_weight):
    
    # lock size to 1024 rather than 512
    max_scls = 5
    sz = 1024

    # swith to 512 if you get an OOM error!
    """""
    max_scls = 4
    sz = 512
    """""

    # could add to args later
    flip_aug = False
    dont_colorize = False
    content_loss = False

    assert (0.0 <= style_weight) and (style_weight <= 1.0), "style weight must be between 0 and 1"

    # Define feature extractor
    cnn = misc.to_device(Vgg16Pretrained())
    phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

    # Load images (woulod be good to use our helpers so this is consistent)
    content_im_orig = misc.to_device(load_path_for_pytorch(content_path, target_size=sz)).unsqueeze(0)
    style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)

    # Run Style Transfer
    torch.cuda.synchronize()
    start_time = time.time()
    output = produce_stylization(content_im_orig, style_im_orig, phi,
                                max_iter=200,
                                lr=2e-3,
                                content_weight=style_weight,
                                max_scls=max_scls,
                                flip_aug=flip_aug,
                                content_loss=content_loss,
                                dont_colorize=dont_colorize)
    torch.cuda.synchronize()

    # Should get anything printing on conda node in vvvv
    print(filename(output_path) + ' done, total time: {}'.format(time.time() - start_time))

    # Convert from pyTorch to numpy, clip to valid range
    new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

    # Save stylized output
    save_im = (new_im_out * 255).astype(np.uint8)
    imwrite(output_path, save_im)

if __name__ == '__main__':

    # Define command line parser and get command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', nargs='+', type=str, help='list of content image paths', default=["inputs/content/C1.png"])
    parser.add_argument('--style_path', nargs='+', type=str, help='list of style image paths', default=["inputs/style/S3.jpg"])
    parser.add_argument('--output_path', nargs='+', type=str, help='list of saved output images paths', default=["inputs/out.jpg"])
    parser.add_argument('--style_weight', nargs='+', type=float, default=[0.75])
    args = parser.parse_args()

    assert torch.cuda.is_available(), "attempted to use gpu when unavailable"

    # Fix Random Seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    cycle_args(style_transfer, *vars(args).values()) 

    # Free gpu memory in case something else needs it later
    torch.cuda.empty_cache()



