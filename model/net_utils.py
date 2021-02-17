import torch
import torch.nn as nn

def construct_psv_grid(an, D, num_source, ind_source, disp_range, N, h, w):
    grid = []
    for k_t in range(0, an*an):
        for step in range(0, D):
            for k_s in range(0, num_source):
                ind_s = ind_source[k_s].type_as(disp_range)
                ind_t = torch.arange(an*an)[k_t].type_as(disp_range)
                ind_s_h = torch.floor(ind_s / an)
                ind_s_w = ind_s % an
                ind_t_h = torch.floor(ind_t / an)
                ind_t_w = ind_t % an
                disp = disp_range[step]

                XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(disp_range)  # [N,h,w]
                YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(disp_range)

                grid_w_t = XX + disp * (ind_t_w - ind_s_w)
                grid_h_t = YY + disp * (ind_t_h - ind_s_h)

                grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
                grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0

                grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [N,h,w,2]
                grid.append(grid_t)

    grid = torch.cat(grid, 0)  # [N*an2*D*4,h,w,2]
    return grid

def construct_syn_grid(an, num_source, ind_source, disp_target, N, h, w):
    grid = []
    for k_t in range(0, an*an):
        for k_s in range(0, num_source):
            ind_s = ind_source[k_s].type_as(disp_target)
            ind_t = torch.arange(an*an)[k_t].type_as(disp_target)
            ind_s_h = torch.floor(ind_s / an)
            ind_s_w = ind_s % an
            ind_t_h = torch.floor(ind_t / an)
            ind_t_w = ind_t % an
            disp = disp_target[:, torch.arange(an*an)[k_t], :, :]

            XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(disp_target)  # [N,h,w]
            YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(disp_target)
            grid_w_t = XX + disp * (ind_t_w - ind_s_w)
            grid_h_t = YY + disp * (ind_t_h - ind_s_h)
            grid_w_t_norm = 2.0 * grid_w_t / (w - 1) - 1.0
            grid_h_t_norm = 2.0 * grid_h_t / (h - 1) - 1.0
            grid_t = torch.stack((grid_w_t_norm, grid_h_t_norm), dim=3)  # [N,h,w,2]
            grid.append(grid_t)

    grid = torch.cat(grid, 0)  # [N*an2*4,h,w,2]
    return grid

class AltFilter(nn.Module):
    def __init__(self, an, ch):
        super(AltFilter, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.spaconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.angconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.an = an
        self.an2 = an * an

    def forward(self, x):
        N, c, h, w = x.shape  # [N*81,c,h,w]
        N = N // self.an2

        out = self.relu(self.spaconv(x))  # [N*81,c,h,w]

        out = out.view(N, self.an2, c, h * w)
        out = torch.transpose(out, 1, 3)  # [N,h*w,c,81]
        out = out.view(N * h * w, c, self.an, self.an)  # [N*h*w,c,9,9]

        out = self.relu(self.angconv(out))  # [N*h*w,c,9,9]

        out = out.view(N, h * w, c, self.an2)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * self.an2, c, h, w)  # [N*81,c,h,w]

        return out


def make_Altlayer(layer_num, an, ch):
    layers = []
    for i in range(layer_num):
        layers.append(AltFilter(an, ch))
    return nn.Sequential(*layers)


class AltFilter_1D(nn.Module):
    def __init__(self, an, ch):
        super(AltFilter_1D, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.spaconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 3), stride=1, padding=2, dilation=2)
        self.angconv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.an = an

    def forward(self, x):
        N, c, h, w = x.shape  # [N*4,c,h,w]
        N = N // self.an

        out = self.relu(self.spaconv(x))  # [N*4,c,h,w]

        out = out.view(N, self.an, c, h * w)
        out = torch.transpose(out, 1, 3)  # [N,h*w,c,4]
        out = out.view(N * h * w, c, self.an, 1)  # [N*h*w,c,4,1]

        out = self.relu(self.angconv(out))  # [N*h*w,c,4,1]

        out = out.view(N, h * w, c, self.an)
        out = torch.transpose(out, 1, 3)
        out = out.view(N * self.an, c, h, w)  # [N*4,c,h,w]

        return out


def make_Altlayer_1D(layer_num, an, ch):
    layers = []
    for i in range(layer_num):
        layers.append(AltFilter_1D(an, ch))
    return nn.Sequential(*layers)
