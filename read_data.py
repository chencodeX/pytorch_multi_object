#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-09-05
Modify Date: 2018-09-05
descirption: ""
'''

import numpy as np
from scipy.io import loadmat


def read_data():
    xm = loadmat('xm.mat')
    ym = loadmat('ym.mat')
    xm1 = loadmat('xm1.mat')
    ym1 = loadmat('ym1.mat')
    phy = loadmat('phy.mat')
    p01 = loadmat('p01.mat')
    locx = loadmat('locx.mat')
    locy = loadmat('locy.mat')
    ampaxy0 = loadmat('ampaxy0.mat')

    xm = np.transpose(xm['xm'])
    ym = np.transpose(ym['ym'])
    xm1 = np.transpose(xm1['xm1'])
    ym1 = np.transpose(ym1['ym1'])
    phy = np.transpose(phy['phy'])
    p01 = np.transpose(p01['p01'])
    locx = locx['locx'].T
    locy = np.transpose(locy['locy'])
    ampaxy0 = np.transpose(ampaxy0['ampaxy0'])
    print locy[:4, :4]
