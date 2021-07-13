
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import os
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from time import time

from utils import dataset, util
from model import model_lfasr

#--------------------------------------------------------------------------#
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Light Field Reconstruction")

#training settings
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=500, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default = 64, help="Training patch size")
parser.add_argument("--batch_size", type=int, default = 1, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--max_epoch", type=int, default=1500, help="maximum epoch for training")
parser.add_argument("--num_cp", type=int, default=25, help="Number of epoches for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epoches for saving loss figure")
parser.add_argument("--smooth", type=float, default=0.001, help="smooth loss weight")
parser.add_argument("--dataset", type=str, default="SIG", help="Dataset for training")
parser.add_argument("--dataset_path", type=str, default="./LFData/train_SIG.h5", help="Dataset file for training")
parser.add_argument("--layer_num", type=int, default=4, help="layer_num of SAS")
parser.add_argument("--psv_range", type=int, default=3, help="depth range for psv")
parser.add_argument("--psv_step", type=int, default=50, help="step number for psv")
parser.add_argument("--arb_sample", type=int, default=1, help="arbitrary input position?")
parser.add_argument("--angular_out", type=int, default=7, help="angular number of the dense light field [AngOut x AngOut]")
parser.add_argument("--angular_in", type=int, default=3, help="angular number of the sparse light field, [AngIn x AngIn](fixed) or AngIn(random)")

opt = parser.parse_args()
print(opt)
#--------------------------------------------------------------------------#

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # model dir
    if opt.arb_sample:
        opt.num_source = opt.angular_in
        model_dir = 'model_arbIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr, opt.step, opt.reduce)
        opt.crop_size = 0

    else:
        opt.num_source = opt.angular_in * opt.angular_in
        model_dir = 'model_fixIn_{}_S{}_pr{}_ps{}_lr{}_step{}x{}'.format(opt.dataset, opt.num_source, opt.psv_range, opt.psv_step, opt.lr,opt.step,opt.reduce)
        opt.crop_size = opt.psv_range * (opt.angular_out - 1)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Data loader
    print('===> Loading datasets')
    train_set = dataset.TrainDataFromHdf5(opt.dataset_path, opt)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))

    # Build model
    print("building net")
    model = model_lfasr.Net_LFASR(opt).to(device)

    # optimizer and loss logger
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)

    # optionally resume from a checkpoint
    if opt.resume_epoch:
        resume_path = join(model_dir,'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            losslogger = checkpoint['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    # training
    print('==>training')
    for epoch in range(opt.resume_epoch + 1, opt.max_epoch):
        model.train()
        loss_count = 0.
        for k in range(10):
            for i, batch in enumerate(train_loader, 1):
                ind_source, input, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)

                disp, inter_lf, pred_lf = model(ind_source, input, opt)
                loss = reconstruction_loss(inter_lf, label) + reconstruction_loss(pred_lf, label) + opt.smooth * smooth_loss(disp)

                loss_count += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scheduler.step()
        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_count/len(train_loader))

        if epoch % opt.num_cp == 0:
            model_save_path = join(model_dir,"model_epoch_{}.pth".format(epoch))
            state = {'epoch':epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'losslogger': losslogger,}
            torch.save(state,model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        if epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'],losslogger['loss'])
            plt.savefig(model_dir+".jpg")
            plt.close()


def reconstruction_loss(X, Y):
    eps = 1e-6
    diff = torch.add(X, -Y)
    error = torch.sqrt( diff * diff + eps )
    loss = torch.sum(error) / torch.numel(error)
    return loss


def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:,:] - pred[:, :, :-1, :]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    loss = 0
    weight = 1.

    dx, dy = gradient(pred_map)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
    return loss


if __name__ == "__main__":
    main()