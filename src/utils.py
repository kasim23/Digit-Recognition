import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import math




def imshow(img):
    """
    Display the input images in a plot.
    
    Parameters:
    img (torch.Tensor): The grid of images to display.
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()
    
    
def init_parameters(params: dict):
    w, b = params['w'], params['b']
    # Function to initialize the weight matrix and the bias term
    nn.init.kaiming_uniform_(params['w'], a=math.sqrt(5))
    if params['b'] is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(params['w'])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(params['b'], -bound, bound)
    
def init_params(params):
    for key in list(params.keys()):
        init_parameters(params[key])