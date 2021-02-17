
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os
from os import listdir
from os.path import join

import math
from math import ceil, floor
import pandas as pd
from PIL import Image
import h5py
from scipy import misc
from skimage.measure import compare_ssim
  
from utils import dataset, util
from model.model_lfasr import Net_LFASR


#----------------------------------------------------------------------------------#
# Test settings
parser = argparse.ArgumentParser(description="PyTorch Light Field Reconstruction Testing")

parser.add_argument("--model_dir", type=str, default="pretrained_models", help="pretrained model dir")
parser.add_argument("--save_dir", type=str, default="results", help="folder to save the test results")
parser.add_argument("--arb_sample", type=int, default=1, help="arbitrary input position or not")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field [AngOut x AngOut]")
parser.add_argument("--angular_in", type=int, default=3, help="angular number of the sparse light field, [AngIn x AngIn](fixed) or AngIn(random)")
parser.add_argument("--layer_num", type=int, default=4, help="layer_num of SAS")
parser.add_argument("--psv_range", type=int, default=3, help="depth range for psv")
parser.add_argument("--psv_step", type=int, default=50, help="step number for psv")
parser.add_argument("--train_dataset", type=str, default="SIG",help="dataset for training")
parser.add_argument("--test_dataset", type=str, default="30scenes",help="dataset for testing")
parser.add_argument("--test_path", type=str, default="./LFData/test_30scenes.h5",help="dataset for testing")
parser.add_argument('--input_ind', action=util.Store_as_array, type=int, nargs='+')
parser.add_argument("--save_img", type=int, default=0,help="save image or not")
parser.add_argument("--crop", type=int, default=0,help="crop the image into patches when out of memory")
opt = parser.parse_args()
print(opt)
#-----------------------------------------------------------------------------------#   


def main():

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model params
    if opt.arb_sample:
        model_name = "model_flexible_{}-{}x{}_pr{}ps{}_{}".format(opt.angular_in, opt.angular_out, opt.angular_out, opt.psv_range, opt.psv_step, opt.train_dataset)
        opt.num_source = opt.angular_in
        opt.crop_size = 0
        opt.test_crop_size = 22
    if not opt.arb_sample:
        model_name = "model_fixed_{}x{}-{}x{}_pr{}ps{}_{}".format(opt.angular_in, opt.angular_in, opt.angular_out, opt.angular_out, opt.psv_range, opt.psv_step, opt.train_dataset)
        opt.num_source = opt.angular_in * opt.angular_in
        opt.crop_size = 22
        opt.test_crop_size = 0

    # Build model
    print("building net")
    model = Net_LFASR(opt).to(device)

    # load pretrained model
    resume_path = join(opt.model_dir, "{}.pth".format(model_name))
    pt_model = torch.load(resume_path)
    pt_dict = pt_model['model']

    model.net_view.load_state_dict(pt_dict, strict=False)
    model.net_refine.load_state_dict(pt_dict, strict=False)
    print('loaded model {}'.format(resume_path))


    # Data loader
    print('===> Loading test datasets')
    test_set = dataset.TestDataFromHdf5(opt.test_path, opt)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
    print('loaded {} LFIs from {}'.format(len(test_loader), opt.test_path))


    # generate save folder
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    opt.save_img_dir = '{}/saveImg_{}_{}_input{}'.format(opt.save_dir, model_name, opt.test_dataset, opt.input_ind)
    if not os.path.exists(opt.save_img_dir):
        os.makedirs(opt.save_img_dir)
    opt.save_csv_name = '{}/res_{}_{}_input{}.csv'.format(opt.save_dir, model_name, opt.test_dataset, opt.input_ind)


    # testing
    print("===> testing")
    model.net_view.eval()
    model.net_refine.eval()

    lf_list = []
    lf_psnr_list = []
    lf_ssim_list = []

    with torch.no_grad():
        for k, batch in enumerate(test_loader):

            input, target_y, lfi_ycbcr = batch[0], batch[1].numpy(), batch[2].numpy()

            input = input.to(device)
            pred_y = predict_y(model, input, opt)

            pred_y = util.crop_boundary(pred_y, opt.test_crop_size)
            pred_y = pred_y.cpu().numpy()

            bd = opt.crop_size + opt.test_crop_size
            target_y = target_y[:, :, bd:-bd, bd:-bd]
            lfi_ycbcr = lfi_ycbcr[:, :, bd:-bd, bd:-bd, :]

            lf_psnr, lf_ssim = save_results(pred_y, lfi_ycbcr, target_y, k, opt)

            lf_list.append(k)
            lf_psnr_list.append(lf_psnr)
            lf_ssim_list.append(lf_ssim)


        dataframe_lfi = pd.DataFrame({'LFI': lf_list, 'psnr Y':lf_psnr_list,'ssim Y':lf_ssim_list})
        dataframe_lfi.to_csv(opt.save_csv_name, index = False, sep=',', mode='a')

        dataframe_lfi = pd.DataFrame({'summary': ['avg'], 'psnr Y':[np.mean(lf_psnr_list)], 'ssim Y':[np.mean(lf_ssim_list)]})
        dataframe_lfi.to_csv(opt.save_csv_name, index=False, sep=',', mode='a')


def predict_y(model, input, opt):
    # coarse view synthesis
    inter_lf = model.net_view(torch.from_numpy(opt.input_ind), input, opt)

    # LF refine
    if not opt.crop:
        pred_y = model.net_refine(inter_lf)
    else:
        length = 180
        crop = 20
        input_l, input_m, input_r = util.CropPatches_w(inter_lf, length, crop)
        pred_l = model.net_refine(input_l)
        pred_m = torch.Tensor(input_m.shape[0], opt.angular_out * opt.angular_out, input_m.shape[2], input_m.shape[3])
        for i in range(input_m.shape[0]):
            pred_m[i:i + 1] = model.net_refine(input_m[i:i + 1])
        pred_r = model.net_refine(input_r)
        pred_y = util.MergePatches_w(pred_l, pred_m, pred_r, inter_lf.shape[2], inter_lf.shape[3], length, crop)  # [N,an2,hs,ws]
    return pred_y


def save_results(pred_y, lfi_ycbcr, target_y, lf_no, opt):

    # save image
    if opt.save_img:
        for i in range(opt.angular_out * opt.angular_out):
            img_ycbcr = lfi_ycbcr[0, i]  # using gt ycbcr for visual results
            img_ycbcr[:, :, 0] = pred_y[0, i]  # [h,w,3]
            img_name = '{}/SynLFI{}_view{}.png'.format(opt.save_img_dir, lf_no, i)
            img_rgb = util.ycbcr2rgb(img_ycbcr)
            img = (img_rgb.clip(0, 1) * 255.0).astype(np.uint8)
            # misc.toimage(img, cmin=0, cmax=255).save(img_name)
            Image.fromarray(img).convert('RGB').save(img_name)

    # compute psnr/ssim
    view_list = []
    view_psnr_y = []
    view_ssim_y = []

    for i in range(opt.angular_out * opt.angular_out):
        if i not in opt.input_ind:
            cur_target_y = target_y[0, i]
            cur_pred_y = pred_y[0, i]

            cur_psnr_y = util.compt_psnr(cur_target_y, cur_pred_y)
            cur_ssim_y = compare_ssim((cur_target_y * 255.0).astype(np.uint8), (cur_pred_y * 255.0).astype(np.uint8), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            view_list.append(i)
            view_psnr_y.append(cur_psnr_y)
            view_ssim_y.append(cur_ssim_y)

    dataframe_lfi = pd.DataFrame({'targetView_LFI{}'.format(lf_no): view_list, 'psnr Y': view_psnr_y, 'ssim Y': view_ssim_y})
    dataframe_lfi.to_csv(opt.save_csv_name, index=False, sep=',', mode='a')

    return np.mean(view_psnr_y), np.mean(view_ssim_y)


if __name__ == '__main__':
    main()