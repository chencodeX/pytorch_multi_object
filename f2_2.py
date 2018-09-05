#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-09-05
Modify Date: 2018-09-05
descirption: ""
'''

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import math
from scipy.io import loadmat
import numpy as np


class F2_2(nn.Module):

    def __init__(self):
        super(F2_2, self).__init__()
        self.f = 2.2e6
        self.w = self.f * 2 * math.pi
        self.length = 25e-3
        self.lam = 0.6e-3
        self.rhom = 1000
        self.cm = 1500
        self.rhoh = 1310
        self.ch = 2110
        self.a = 12.5e-3
        self.NX = 128
        self.xn = 126
        self.yn = 126
        self.delxy = 2 * self.a / self.xn
        self.dlam = 0.2e-3
        self.delk = 1 / self.delxy / self.xn / 3
        self.km = self.w / self.cm
        self.kh = self.w / self.ch

    def init_parms(self):
        xm = loadmat('xm.mat')
        ym = loadmat('ym.mat')
        xm1 = loadmat('xm1.mat')
        ym1 = loadmat('ym1.mat')
        phy = loadmat('phy.mat')
        p01 = loadmat('p01.mat')
        locx = loadmat('locx.mat')
        locy = loadmat('locy.mat')
        ampaxy0 = loadmat('ampaxy0.mat')
        temp1 = loadmat('temp.mat')
        self.xm = np.transpose(xm['xm'])
        self.ym = np.transpose(ym['ym'])
        self.xm1 = np.transpose(xm1['xm1'])
        self.ym1 = np.transpose(ym1['ym1'])
        self.phy = np.transpose(phy['phy'])
        self.p01 = np.transpose(p01['p01'])
        self.locx = locx['locx'].T
        self.locy = np.transpose(locy['locy'])
        self.ampaxy0 = np.transpose(ampaxy0['ampaxy0'])
        self.temp1 = np.transpose(temp1['tmp'])

    def forward(self, m0):
        T = 5e-4 * m0
        xmm = np.round(self.xm / self.dlam + 64.5)
        ymm = np.round(self.ym / self.dlam + 64.5)
        print(xmm)
        print(type(xmm))
        print(self.phy[xmm[0]][:,ymm[0]])
        phyxy = self.phy[xmm][:,ymm] + (self.km - self.kh) * T[xmm][:ymm]
        p0 = self.ampaxy0 * torch.exp(1j * phyxy)

        self.p01[self.xn:self.xn * 2, self.yn:self.yn * 2] = p0
        pk01 = torch.fft(self.p01, 2)
        pk01 = np.fft.ifftshift(pk01)
        pkzl1 = pk01 * self.temp1
        pkzl1 = np.fft.fftshift(pkzl1)
        pzl1 = torch.ifft(pkzl1,2)

        pzl1 = F.avg_pool2d(pzl1,kernel_size=3,stride=1)

        p2 = pzl1[range(139,240),[189]*101]

        return p2



