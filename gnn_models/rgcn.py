'''
Author: your name
Date: 2021-05-12 13:00:03
LastEditTime: 2021-05-12 13:00:43
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /dglke/gnn_models/rgcn.py
'''

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.nn.pytorch import RelGraphConv
