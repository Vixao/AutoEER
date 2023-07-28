import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from utils import drop_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Cell(nn.Module):

  def __init__(self, genotype, C):
    super(Cell, self).__init__()
    #print(C_prev_prev, C_prev, C)
    self.preprocess0 = Identity()
    #print(genotype.normal[i])
    op_names, indices = zip(*genotype.normal)
    concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat)

  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = (len(op_names)+1) // 2
    self._concat = concat
    self._ops = nn.ModuleList()

    for name, index in zip(op_names, indices):
      if 'dgcn' in name:
        op = OPS[name](C)
      else:
        op = OPS[name](C)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0,  drop_prob):
    s0 = self.preprocess0(s0)
    states = [s0]
    for i in range(self._steps):
      if i == 0:
        h1 = states[self._indices[0]]
        op1 = self._ops[0]
        h1 = op1(h1)
        h2 = 0
      else:
        h1 = states[self._indices[2*i-1]]
        h2 = states[self._indices[2*i]]
        op1 = self._ops[2*i-1]
        op2 = self._ops[2*i]
        h1 = op1(h1)
        h2 = op2(h2)
        if self.training and drop_prob > 0.:
          if not isinstance(op1, Identity):
            h1 = drop_path(h1, drop_prob)
          if not isinstance(op2, Identity):
            h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states =states + [s]
    return states[-1]

class NetworkDEAP(nn.Module):

  def __init__(self, C, num_classes, layers, genotype):
    super(NetworkDEAP, self).__init__()
    self._layers = layers

    C_curr = C
    self.stem = nn.Sequential(
      nn.Conv2d(32, C_curr, 1, padding=0, bias=False),#DEAP 32
      nn.BatchNorm2d(C_curr)
    )
    
    C_prev = C_curr
    self.cells = nn.ModuleList()
    self.skip_connect = nn.ModuleList()
    for i in range(layers):
      cell = Cell(genotype, C_curr)
      self.cells += [cell]
    self.skip_connect.append(nn.Conv1d(C_prev, C * 8, (1, 1)))
    #print("cell over")
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev*8, num_classes)

  def forward(self, input):
    #print("input.shape",input.shape)
    s0 = self.stem(input)
    #print("s0.shape",s0.shape)
    skip = 0
    for i, cell in enumerate(self.cells):
      s0 = cell(s0, self.drop_path_prob)
    skip = self.skip_connect[0](s0) + skip
    state = torch.max(F.relu(skip), dim=-1, keepdim=True)[0]
    out = self.global_pooling(state)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

