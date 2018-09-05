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
import numpy as np

from f2_2 import F2_2


m0 = torch.zeros([128, 128], dtype=torch.int32)