import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import cv2
from scipy import misc
from math import ceil
import random

from utils.util import crop_boundary


class TrainDataFromHdf5(data.Dataset):
    def __init__(self, file_path, opt):
        super(TrainDataFromHdf5, self).__init__()
        
        hf = h5py.File(file_path)
        self.LFI = hf.get('LFI')  # [N,ah,aw,h,w]
   
        self.psize = opt.patch_size
        self.arb_sample = opt.arb_sample
        self.ang_out = opt.angular_out
        self.ang_in = opt.angular_in
        self.crop = opt.crop_size
    
    def __getitem__(self, index):
                        
        # get one item
        lfi = self.LFI[index]  # [ah,aw,h,w]

        # crop to patch
        H = lfi.shape[2]
        W = lfi.shape[3]

        x = random.randrange(0, H-self.psize)    
        y = random.randrange(0, W-self.psize) 
        lfi = lfi[:self.ang_out, :self.ang_out, x:x+self.psize, y:y+self.psize]  # [ah,aw,ph,pw]
        
        # 4D augmentation
        # flip
        if np.random.rand(1)>0.5:
            lfi = np.flip(np.flip(lfi, 0), 2)
        if np.random.rand(1)>0.5:
            lfi = np.flip(np.flip(lfi, 1), 3)
        # rotate
        r_ang = np.random.randint(1, 5)
        lfi = np.rot90(lfi, r_ang, (2, 3))
        lfi = np.rot90(lfi, r_ang, (0, 1))
            
        
        ##### get input index ######
        if self.arb_sample:
            ind_source = np.array(random.sample(list(np.arange(self.ang_out*self.ang_out)), self.ang_in))
            ind_source = np.sort(ind_source)
            
        else:
            ind_all = np.arange(self.ang_out*self.ang_out).reshape(self.ang_out, self.ang_out)
            delt = (self.ang_out-1) // (self.ang_in-1)
            ind_source = ind_all[0:self.ang_out:delt, 0:self.ang_out:delt]
            ind_source = ind_source.reshape(-1)
            
        ##### get input and label ######
        lfi = lfi.reshape(-1, self.psize, self.psize)  # [ah*aw,ph,pw]
        label = torch.from_numpy(lfi.astype(np.float32)/255.0)  # [an2,h,w]
        label = crop_boundary(label, self.crop)

        input = lfi[ind_source, :, :]  # [num_source,ph,pw]
        input = torch.from_numpy(input.astype(np.float32)/255.0)  # [num_source,h,w]

        return ind_source, input, label

    def __len__(self):
        return self.LFI.shape[0]


class TestDataFromHdf5(data.Dataset):
    def __init__(self, file_path, opt):
        super(TestDataFromHdf5, self).__init__()

        hf = h5py.File(file_path)
        self.LFI_ycbcr = hf.get('LFI_ycbcr')  # [N,ah,aw,h,w,3]

        self.ang_out = opt.angular_out
        self.input_ind = opt.input_ind

    def __getitem__(self, index):
        H, W = self.LFI_ycbcr.shape[3:5]

        lfi_ycbcr = self.LFI_ycbcr[index]  # [ah,aw,h,w,3]
        lfi_ycbcr = lfi_ycbcr[:self.ang_out, :self.ang_out, :].reshape(-1, H, W, 3)  # [ah*aw,h,w,3]

        input = lfi_ycbcr[self.input_ind, :, :, 0]  # [num_source,H,W]
        target_y = lfi_ycbcr[:, :, :, 0]  # [ah*aw,h,w]

        input = torch.from_numpy(input.astype(np.float32) / 255.0)
        target_y = torch.from_numpy(target_y.astype(np.float32) / 255.0)

        # keep cbcr for RGB reconstruction (Using Ground truth just for visual results)
        lfi_ycbcr = torch.from_numpy(lfi_ycbcr.astype(np.float32) / 255.0)

        return input, target_y, lfi_ycbcr

    def __len__(self):
        return self.LFI_ycbcr.shape[0]


