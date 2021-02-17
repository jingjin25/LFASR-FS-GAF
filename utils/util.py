
import torch
import torch.nn as nn
import torch.nn.functional as functional
import math
import argparse
import numpy as np
import copy


def warping(disp, ind_source, ind_target, img_source, an):
    '''warping one source image/map to the target'''

    # disp:       [scale] or [N,h,w]
    # ind_souce:  (int)
    # ind_target: (int)
    # img_source: [N,h,w]
    # an:         angular number
    # ==> out:    [N,1,h,w]

    N, h, w = img_source.shape
    ind_source = ind_source.type_as(disp)
    ind_target = ind_target.type_as(disp)

    # coordinate for source and target
    ind_h_source = torch.floor(ind_source / an)
    ind_w_source = ind_source % an

    ind_h_target = torch.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
    YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)

    grid_w = XX + disp * (ind_w_target - ind_w_source)
    grid_h = YY + disp * (ind_h_target - ind_h_source)

    grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0
    grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0

    grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)  # [N,h,w,2]

    # inverse warp
    img_source = torch.unsqueeze(img_source, 0)
    img_target = functional.grid_sample(img_source, grid)  # [N,1,h,w]
    img_target = torch.squeeze(img_target, 1)  # [N,h,w]

    return img_target


def crop_boundary(I, crop_size):
    '''crop the boundary (the last 2 dimensions) of a tensor'''
    if crop_size == 0:
        return I

    if crop_size > 0:
        size = list(I.shape)
        I_crop = I.view(-1, size[-2], size[-1])
        I_crop = I_crop[:, crop_size:-crop_size, crop_size:-crop_size]
        size[-1] -= crop_size * 2
        size[-2] -= crop_size * 2
        I_crop = I_crop.view(size)
        return I_crop

class Store_as_array(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        values = np.array(values)
        return super().__call__(parser, namespace, values, option_string)




def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16. / 255.
    rgb[:,1:] -= 128. / 255.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 1).reshape(shape).astype(np.float32)

def CropPatches_w(image,len,crop):
    #image [1,4,ph,pw]
    #left [1,4,h,lw]
    #middles[n,4,h,mw]
    #right [1,4,h,rw]
    an,h,w = image.shape[1:4]
    left = image[:,:,:,0:len+crop]
    num = math.floor((w-len-crop)/len)
    middles = torch.Tensor(num,an,h,len+crop*2).to(image.device)
    for i in range(num):
        middles[i] = image[0,:,:,(i+1)*len-crop:(i+2)*len+crop]
    right = image[:,:,:,-(len+crop):]
    return left,middles,right

def MergePatches_w(left,middles,right,h,w,len,crop):
    #[N,4,h,w]
    n,a = left.shape[0:2]
    out = torch.Tensor(n,a,h,w).to(left.device)
    out[:,:,:,:len] = left[:,:,:,:-crop]
    for i in range(middles.shape[0]):
        out[:,:,:,len*(i+1):len*(i+2)] = middles[i:i+1,:,:,crop:-crop]
    out[:,:,:,-len:]=right[:,:,:,crop:]
    return out


def compt_psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))