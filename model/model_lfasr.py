
import torch
import torch.nn as nn
import torch.nn.functional as functional

import sys
sys.path.insert(0,'../utils/')
from utils.util import warping, crop_boundary
from model.net_utils import make_Altlayer, construct_psv_grid, construct_syn_grid


class Net_view(nn.Module):
    '''coarse SAI synthesis network'''

    def __init__(self, opt):
        super(Net_view, self).__init__()

        self.conv_perPlane = nn.Sequential(
            nn.Conv2d(opt.num_source, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.conv_crossPlane = nn.Sequential(
            nn.Conv2d(4 * opt.psv_step, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(200, opt.psv_step, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.conv_disp = nn.Sequential(
            nn.Conv2d(opt.psv_step, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1 + opt.num_source, kernel_size=3, stride=1, padding=1),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, ind_source, img_source, opt):

        an = opt.angular_out
        an2 = opt.angular_out * opt.angular_out

        N, num_source, h, w = img_source.shape  # [N,4,h,w]
        ind_source = torch.squeeze(ind_source)  # [4]

        h_c = h - 2 * opt.crop_size
        w_c = w - 2 * opt.crop_size

        D = opt.psv_step
        disp_range = torch.linspace(-1 * opt.psv_range, opt.psv_range, steps=D).type_as(img_source)  # [D]

        if self.training:
            # PSV
            psv_input = img_source.view(N * num_source, 1, h, w).repeat(D * an2, 1, 1, 1)  # [N*an2*D*4,1,h,w]
            grid = construct_psv_grid(an, D, num_source, ind_source, disp_range, N, h, w)  # [N*an2*D*4,h,w,2]
            PSV = functional.grid_sample(psv_input, grid).view(N, an2, D, num_source, h, w)  # [N*an2*D*4,1,h,w]-->[N,an2,D,4,h,w]
            PSV = crop_boundary(PSV, opt.crop_size)

            # disparity & confidence estimation
            perPlane_out = self.conv_perPlane(PSV.view(N * an2 * D, num_source, h_c, w_c))  # [N*an2*D,4,h,w]
            crossPlane_out = self.conv_crossPlane(perPlane_out.view(N * an2, D * 4, h_c, w_c))  # [N*an2,D,h,w]
            disp_out = self.conv_disp(crossPlane_out)  # [N*an2,5,h,w]
            disp_target = disp_out[:, 0, :, :].view(N, an2, h_c, w_c)  # disparity for each view
            disp_target = functional.pad(disp_target, pad=[opt.crop_size, opt.crop_size, opt.crop_size, opt.crop_size], mode='constant', value=0)
            conf_source = disp_out[:, 1:, :, :].view(N, an2, num_source, h_c, w_c)  # confidence of source views for each view
            conf_source = self.softmax(conf_source)

            # intermediate LF
            warp_img_input = img_source.view(N * num_source, 1, h, w).repeat(an2, 1, 1, 1)  # [N*an2*4,1,h,w]
            grid = construct_syn_grid(an, num_source, ind_source, disp_target, N, h, w) # [N*an2*4,h,w,2]
            warped_img = functional.grid_sample(warp_img_input, grid).view(N, an2, num_source, h, w)  # {N,an2,4,h,w]
            warped_img = crop_boundary(warped_img, opt.crop_size)

            inter_lf = torch.sum(warped_img * conf_source, dim=2)  # [N,an2,h,w]
            return disp_target, inter_lf

        else:
            inter_lf = torch.zeros((N, an2, h_c, w_c)).type_as(img_source)
            for k_t in range(0, an2):  # for each target view
                ind_t = torch.arange(an2)[k_t]

                # disparity & confidence estimation
                PSV = torch.zeros((N, D, num_source, h, w)).type_as(img_source)
                for step in range(0, D):
                    for k_s in range(0, num_source):
                        ind_s = ind_source[k_s]
                        disp = disp_range[step]
                        PSV[:, step, k_s] = warping(disp, ind_s, ind_t, img_source[:, k_s], an)

                PSV = crop_boundary(PSV, opt.crop_size)

                perPlane_out = self.conv_perPlane(PSV.view(N * D, num_source, h_c, w_c))  # [N*D,4,h,w]
                crossPlane_out = self.conv_crossPlane(perPlane_out.view(N, D * 4, h_c, w_c))  # [N,D,h,w]
                disp_out = self.conv_disp(crossPlane_out)  # [N,5,h,w]
                disp_target = disp_out[:, 0, :, :]  # [N,h,w] disparity for each view
                disp_target = functional.pad(disp_target, pad=[opt.crop_size, opt.crop_size, opt.crop_size, opt.crop_size], mode='constant', value=0)
                conf_source = disp_out[:, 1:, :, :]  # [N,4,h_c,w_c] confidence of source views for each view
                conf_source_norm = self.softmax(conf_source)

                # warping source views
                warped_img = torch.zeros(N, num_source, h, w).type_as(img_source)
                for k_s in range(0, num_source):
                    ind_s = ind_source[k_s]
                    disp = disp_target
                    warped_img[:, k_s] = warping(disp, ind_s, ind_t, img_source[:, k_s], an)
                warped_img = crop_boundary(warped_img, opt.crop_size)

                inter_view = torch.sum(warped_img * conf_source_norm, dim=1)  # [N,h,w]
                inter_lf[:, k_t] = inter_view

            return inter_lf


class Net_refine(nn.Module):
    '''efficient LF refinement network'''

    def __init__(self, opt):
        super(Net_refine, self).__init__()

        self.lf_conv0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.lf_altblock = make_Altlayer(layer_num=opt.layer_num, an=opt.angular_out, ch=64)
        self.lf_res_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, inter_lf):
        N, an2, h, w = inter_lf.shape

        feat = self.lf_conv0(inter_lf.view(N * an2, 1, h, w))  # [N*an2,64,h,w]
        feat = self.lf_altblock(feat)  # [N*an2,64,h,w]
        res = self.lf_res_conv(feat).view(N, an2, h, w)  # [N*an2,1,h,w]-->[N,an2,h,w]

        lf = inter_lf + res  # [N,an2,h,w]

        return lf



class Net_LFASR(nn.Module):
    '''end-to-end LF reconstruction network'''

    def __init__(self, opt):
        super(Net_LFASR, self).__init__()

        self.net_view = Net_view(opt)
        self.net_refine = Net_refine(opt)

    def forward(self, ind_source, img_source, opt):

        disp_lf, inter_lf = self.net_view(ind_source, img_source, opt)
        lf = self.net_refine(inter_lf)

        return [disp_lf, inter_lf, lf]
