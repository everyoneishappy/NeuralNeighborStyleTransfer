

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


# Fix Random Seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# These should come from our helpers
################################################################################

# foreach(max) style for vvvv.  
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



def style_transfer(content_path, style_path, output_path, style_weight):
    
    # lock size to 1024 rather than 512
    max_scls = 5
    sz = 1024

    # could add to args later
    flip_aug = False
    dont_colorize = False
    content_loss = False

    assert (0.0 <= style_weight) and (style_weight <= 1.0), "style weight must be between 0 and 1"


    # Define feature extractor
    cnn = misc.to_device(Vgg16Pretrained())
    phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

    # Load images
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
    print(filename(output_path) + ' done, total time: {}'.format(time.time() - start_time))

    # Convert from pyTorch to numpy, clip to valid range
    new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

    # Save stylized output
    save_im = (new_im_out * 255).astype(np.uint8)
    imwrite(output_path, save_im)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', nargs='+', type=str, help='list of content image paths', default=["inputs/content/C1.png"])
    parser.add_argument('--style_path', nargs='+', type=str, help='list of style image paths', default=["inputs/style/S3.jpg"])
    parser.add_argument('--output_path', nargs='+', type=str, help='list of saved output images paths', default=[""])
    parser.add_argument('--style_weight', nargs='+', type=float, default=[0.75])

    assert torch.cuda.is_available(), "attempted to use gpu when unavailable"

    args = parser.parse_args()
    cycle_args(style_transfer, *vars(args).values()) 

    torch.cuda.empty_cache()



    
""" 
# Define command line parser and get command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--content_path'   , type=str, default=None, required=True)
parser.add_argument('--style_path'     , type=str, default=None, required=True)
parser.add_argument('--output_path'    , type=str, default=None, required=True)
parser.add_argument('--high_res'       , action='store_true'                  )
parser.add_argument('--cpu'            , action='store_true'                  )
parser.add_argument('--no_flip'        , action='store_true'                  )
parser.add_argument('--content_loss'   , action='store_true'                  )
parser.add_argument('--dont_colorize'  , action='store_true'                  )
parser.add_argument('--alpha'          , type=float, default=0.75             )
args = parser.parse_args()

# Interpret command line arguments
content_path = args.content_path
style_path = args.style_path
output_path = args.output_path
max_scls = 4
sz = 512
if args.high_res:
    max_scls = 5
    sz = 1024
flip_aug = (not args.no_flip)
content_loss = args.content_loss
misc.USE_GPU = (not args.cpu)
content_weight = 1. - args.alpha

# Error checking for arguments
# error checking for paths deferred to imageio
assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

# Define feature extractor
cnn = misc.to_device(Vgg16Pretrained())
phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

# Load images
content_im_orig = misc.to_device(load_path_for_pytorch(content_path, target_size=sz)).unsqueeze(0)
style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)

# Run Style Transfer
torch.cuda.synchronize()
start_time = time.time()
output = produce_stylization(content_im_orig, style_im_orig, phi,
                            max_iter=200,
                            lr=2e-3,
                            content_weight=content_weight,
                            max_scls=max_scls,
                            flip_aug=flip_aug,
                            content_loss=content_loss,
                            dont_colorize=args.dont_colorize)
torch.cuda.synchronize()
print('Done! total time: {}'.format(time.time() - start_time))

# Convert from pyTorch to numpy, clip to valid range
new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

# Save stylized output
save_im = (new_im_out * 255).astype(np.uint8)
imwrite(output_path, save_im)

# Free gpu memory in case something else needs it later
if misc.USE_GPU:
    torch.cuda.empty_cache() """