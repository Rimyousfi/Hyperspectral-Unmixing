import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
dtype = torch.float32
import scipy.io
import scipy.linalg



def Eucli_dist(v1, v2):
    """
    Calculates the Euclidean distance between two vectors v1 and v2.
    
    Args:
        v1: The first vector, either an array or a list of coordinates.
        v2: The second vector, either an array or a list of coordinates.
    
    Returns:
        The Euclidean distance between v1 and v2.
        
    Raises:
        ValueError: If v1 and v2 do not have the same dimension.
    """
    if len(v1) != len(v2):
        raise ValueError("The vectors must have the same dimension.")
    return np.sqrt(np.sum((np.array(v1) - np.array(v2)) ** 2))



def Endmember_extract(x, p):
    """
    Extracts 'p' endmembers from the given hyperspectral image data, as the most representative pure endmembers by selecting the most distinct points in the data.
    
    Args:
        x: numpy array of shape (D, N) where:
            - D is the number of spectral bands.
            - N is the number of pixels (observations).
        p: Number of endmembers to extract.

    Returns:
        I: Indices of the selected endmembers.
        d: Distance matrix used during selection.
    """

    # Get dimensions: D (number of spectral bands), N (number of pixels)
    [D, N] = x.shape

    # Initialization
    Z1 = np.zeros((1, 1))  
    O1 = np.ones((1, 1))  
    d = np.zeros((p, N))   
    I = np.zeros((p, 1))   
    V = np.zeros((1, N))   
    ZD = np.zeros((D, 1))  

    # Find the farthest point from the origin (first endmember)
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), ZD)  

    I = np.argmax(d[0, :])  # Select the pixel farthest from the origin

    # Compute distances from the first selected endmember
    for i in range(N):
        d[0, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I].reshape(D, 1))

    # Iteratively select the next (p-1) endmembers
    for v in range(1, p):
        D1 = np.concatenate((d[0:v, I].reshape((v, I.size)), np.ones((v, 1))), axis=1)
        D2 = np.concatenate((np.ones((1, v)), Z1), axis=1)
        D4 = np.concatenate((D1, D2), axis=0)
        D4 = np.linalg.pinv(D4)  

        # Compute variance for each pixel and select the one maximizing variance
        for i in range(N):
            D3 = np.concatenate((d[0:v, i].reshape((v, 1)), O1), axis=0)
            V[0, i] = np.dot(np.dot(D3.T, D4), D3)[0][0]  # Compute variance measure
        # Select the pixel with the highest variance as the next endmember
        I = np.append(I, np.argmax(V))

        # Update distance matrix with the new endmember
        for i in range(N):
            d[v, i] = Eucli_dist(x[:, i].reshape(D, 1), x[:, I[v]].reshape(D, 1))

    # Step 4: Sort the selected endmembers
    per = np.argsort(I)  
    I = np.sort(I)       
    d = d[per, :]        
    return I, d







def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero', downsample_mode='stride'):
    """
    Builds and return a sequence of layers for performing a convolution operation (a convolution block)

    Params:
       in_f (int): The number of input feature channels 
       out_f (int): The number of output feature channels 
       kernel_size (int or tuple): The size of the convolutional kernel (filter)
       stride (int, optional): The step size by which the kernel moves across the input
       bias (bool, optional): Whether or not to include a bias term in the convolution
       pad (str, optional): The padding type to be applied to the input before the convolution
       downsample_mode (str, optional): The mode used to downsample the input when the stride is greater than 1
       
    
    """
    downsampler = None
    if stride != 1 and downsample_mode != 'stride':

        if downsample_mode == 'avg':
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == 'max':
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode  in ['lanczos2', 'lanczos3']:
            downsampler = Downsampler(n_planes=out_f, factor=stride, kernel_type=downsample_mode, phase=0.5, preserve_size=True)
        else:
            assert False

        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
  
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)


    layers = filter(lambda x: x is not None, [padder, convolver, downsampler])
    return nn.Sequential(*layers)



def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params


