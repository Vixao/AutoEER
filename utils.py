import os
import numpy as np
import torch
import shutil
import scipy.io as sio
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics import f1_score
from scipy import stats
        
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):

  batch_size = target.size(0)


  _, pred = output.topk(1, 1, True, True)

  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))


  res = []
  for k in topk:
    correct_k = correct[:1].contiguous().view(-1).float().sum(0)

    res.append(correct_k.mul_(1.0/batch_size))
  return res

def f1(output, target):

  batch_size = target.size(0)
  _, pred = output.topk(1, 1, True, True)
  pred = pred.t().view(-1)
  pred = pred.cpu().numpy()
  target = target.cpu().numpy()
  score=f1_score(target, pred, average='macro')

  return score




def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)


def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.05, last_epoch=-1):
        self.total_epochs = total_epochs
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        # if self.curr_temp < self.temp_min:
        #     self.curr_temp = self.temp_min

        self.curr_temp = max(self.base_temp * 0.90 ** self.last_epoch, self.temp_min)

        return self.curr_temp


def average_pearsonr(x1, x2):

    avg_r = 0
    #print(x1.shape)
    x1 = x1.reshape(-1,16)
    x2 = x2.reshape(-1,16)
    for i in range(x1.shape[0]):
        r, _ = stats.pearsonr(x1[i], x2[i])
        avg_r += r
    avg_r /= x1.shape[0]
    return avg_r

def get_adj_matrix(data):
  data = data.transpose([0,3,1,2]).astype('float64')
  A = np.zeros((data.shape[-2],data.shape[-2]))
  for i in range(data.shape[-2]):
      for j in range(i+1, data.shape[-2]):
          #print("data[:, :, i].shape",data[:, :, i].shape)
          avg_r = abs(average_pearsonr(data[:, :, i], data[:, :, j]))
          A[i][j] = avg_r
          A[j][i] = avg_r
  return A

def exp_mask_for_high_rank(val, val_mask): 
    # Expand the mask dimension to match the value dimension 
    val_mask = val_mask.unsqueeze(-1) 
    # Create a very negative number 
    very_negative_number = -torch.finfo(val.dtype).max 
    # Add the very negative number to the masked values 
    return val + (1 - val_mask.to(val.dtype)) * very_negative_number