import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import operator
import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class MixedOp(nn.Module):

  def __init__(self, C):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.mp = nn.MaxPool2d(2,2)
    self.k = 4
    #print("channel proportion",self.k)
    for primitive in PRIMITIVES:
      if 'dgcn' in primitive:
        op = OPS[primitive](C //self.k)
      else:
        op = OPS[primitive](C // self.k)
      self._ops.append(op)


  def forward(self, x, weights):
    #channel proportion k=4  
    dim_2 = x.shape[1]
    xtemp = x[ : , : dim_2//self.k, :, :]
    xtemp2 = x[ : ,  dim_2//self.k:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    ans = torch.cat([temp1,xtemp2],dim=1)
    ans = channel_shuffle(ans,self.k)
    return ans

class Cell(nn.Module):

  def __init__(self, steps,  C):
    super(Cell, self).__init__()

    self.preprocess0 = Identity()
    self._steps = steps

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(1 + i):
        op = MixedOp(C)
        self._ops.append(op)

  def forward(self, s0, weights,weights2):
    s0 = self.preprocess0(s0)

    states = [s0]
    offset = 0
    for i in range(self._steps):
      s = sum(weights2[offset+j]*self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return states[-1]


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, temp, steps=4):
    super(Network, self).__init__()
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._temp = temp

    C_curr = C
    #For SEED, 32 needs to be changed to 62
    self.stem = nn.Sequential(
      nn.Conv2d(32, C_curr, 1, padding=0, bias=False),
      nn.BatchNorm2d(C_curr)
    )

    C_prev = C_curr
    self.cells = nn.ModuleList()
    self.skip_connect = nn.ModuleList()
    for i in range(layers):
      cell = Cell(self._steps, C_curr)
      self.cells += [cell]
    self.skip_connect.append(nn.Conv1d(C_prev, C * 8, (1, 1)))

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev*8, num_classes)

    self._initialize_alphas()

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self._temp).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):

    s0 = self.stem(input)
    skip = 0
    prev = [s0]
    n2 = 1  # for inter-cell
    start2 = 0
    for i, cell in enumerate(self.cells):
      weights = F.softmax(self.alphas / self._temp, dim=-1)
      n = 2
      start = 1
      weights2 = F.softmax(self.betas[0:1], dim=-1)
      start2 += n2
      n2 += 1
      for j in range(self._steps-1):
        end = start + n
        tw2 = F.softmax(self.betas[start:end], dim=-1)
        start = end
        n += 1
        weights2 = torch.cat([weights2,tw2],dim=0)
    
      s0 = cell(s0, weights,weights2)
   
      #prev.append(s0)
    skip = self.skip_connect[0](s0) + skip
      

    state = torch.max(F.relu(skip), dim=-1, keepdim=True)[0]
    out = self.global_pooling(state)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits

  def _loss(self, input, target):
    logits = self(input)
    return self._criterion(logits, target.long()) 

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(1 + i))
    num_ops = len(PRIMITIVES)
    np.random.seed(2)
    self.alphas = Variable(1e-3 * torch.randn( k, num_ops).to(DEVICE), requires_grad=True)
    self.betas = Variable(1e-3 * torch.randn( k).to(DEVICE), requires_grad=True)
    self._arch_parameters = [self.alphas, self.betas]
    

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights,weights2):
      gene = []
      n = 1
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        W2 = weights2[start:end].copy()
        for j in range(n):
          W[j,:]=W[j,:]*W2[j]
        #print('W',W)
        max_edge = sorted(
              range(i),
              key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:1]
        #print(sorted(
        #      range(i),
        #      key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none'))))
        #print("max_edge",max_edge)
        edges = max_edge + [i]
        
        #edges = sorted(range(i + 2), key=lambda x: -W2[x])[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene
    
    n = 2
    start = 1
    weights2 = F.softmax(self.betas[0:1], dim=-1)
    for _ in range(self._steps - 1):
      end = start + n
      tw2 = F.softmax(self.betas[start:end], dim=-1)
      start = end
      n += 1
      weights2 = torch.cat([weights2, tw2], dim=0)
    #print("genotype weights2",weights2)
    gene = _parse(F.softmax(self.alphas, dim=-1).data.cpu().numpy(),
                               weights2.data.cpu().numpy())

    concat = range(1 + self._steps - 4, self._steps + 1)
    genotype = Genotype(gene, concat)
    return genotype


