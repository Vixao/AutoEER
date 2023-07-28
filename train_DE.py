import os
import ast
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split

from torch.autograd import Variable
from model import NetworkDEAP as Network


parser = argparse.ArgumentParser("deap")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=7, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PCDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--flag', type=str, default='v', help='valence or arousal')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

label_CLASSES = 2

class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  
  subject_name = ['01','02', '03', '04','05', '06', '07', '08',
              '09', '10','11', '12', '13', '14', '15','16','17', '18',
              '19', '20', '21', '22', '23', '24', '25','26','27','28','29','30','31','32']

  for i in range(len(subject_name)):
      print("\nprocessing: ", subject_name[i], "......")
      name=subject_name[i]
      num=0
      with open(args.flag+"_arch.txt","r",encoding="utf-8") as f:
            for userline in f.readlines():
                if num==i:
                  userline=eval(userline)
                  print(userline)
                  genotype=eval("genotypes.%s" % userline[name])
                  #print(genotype)
                  break
                else:
                  num=num+1
      
      #DEAP
      dataset_dir = "/home/wyx/data/DEAP_DE/3D_2400_32_4_4/"
      print("loading ",dataset_dir+"DE_s"+name,".mat")
      data_file = sio.loadmat(dataset_dir+"DE_s"+name+".mat")
      data= data_file["data"].astype(np.float32)
      if args.flag == 'a':
        label=np.squeeze(data_file["arousal_labels"].transpose())
      elif args.flag == 'v':
        label=np.squeeze(data_file["valence_labels"].transpose())
      else:
        label=np.squeeze(data_file["av_labels"].transpose())
      label=np.array(label, dtype=np.int32)
      '''
      #SEED
      X = np.load('/home/wyx/data/SEED/X_3D.npy')
      y = np.load('/home/wyx/data/SEED/y_3D.npy')
      data = X[i * 3:i * 3 + 3]
      label=y[i * 10182:i * 10182 + 10182]
      data=data.reshape(-1,62,4,4).astype(np.float32)
      '''
      datas,labels={},{}
      datas["train"], datas["val"], labels["train"], labels["val"] = train_test_split(data, label, 
                                      test_size=0.2, 
                                      stratify=label,
                                      shuffle=True,
                                      random_state=12)
      
      model = Network(args.init_channels, label_CLASSES, args.layers, genotype)
      model = model.cuda()

      logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

      criterion = nn.CrossEntropyLoss()
      criterion = criterion.cuda()
      """
      optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay) 
      """
      optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
      
      torch_data_train = GetLoader(datas["train"], labels["train"])
      torch_data_test = GetLoader(datas["val"], labels["val"])
      train_queue = torch.utils.data.DataLoader(
        torch_data_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

      valid_queue = torch.utils.data.DataLoader(
        torch_data_test, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

      #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
      best_acc = 0.0
      best_macro_f1 = 0.0
      log_dir = "logs/train/deap/"+args.flag+"/"+name
      if not os.path.exists(log_dir):
          os.makedirs(log_dir)
      writer = SummaryWriter(log_dir)
      for epoch in range(args.epochs):
            #scheduler.step()
            #lr = scheduler.get_lr()[0]
            logging.info('epoch %d  lr %e', epoch, optimizer.param_groups[0]['lr'])
            model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

            train_acc, train_obj, train_macro_f1 = train(train_queue, model, criterion, optimizer)
            logging.info('train_acc %f, train_macro_f1 %f', train_acc, train_macro_f1)
            writer.add_scalar('Train/Loss', train_obj, epoch )
            writer.add_scalar('Train/Accuracy', train_acc, epoch )
            writer.add_scalar('Train/F1', train_macro_f1, epoch )
            valid_acc, valid_obj, valid_macro_f1 = infer(valid_queue, model, criterion)
            writer.add_scalar('Valid/Loss', valid_obj, epoch )
            writer.add_scalar('Valid/Accuracy', valid_acc, epoch )
            writer.add_scalar('Valid/F1', valid_macro_f1, epoch )
            if valid_acc > best_acc:
                best_acc = valid_acc
                path = os.path.join(args.save, name)
                if not os.path.exists(path):
                    os.makedirs(path)
                utils.save(model, os.path.join(path, 'weights.pt'))
            if valid_macro_f1 > best_macro_f1:
                best_macro_f1 = valid_macro_f1
            logging.info('valid_macro_f1 %f,  valid_acc %f, best_macro_f1 %f, best_acc %f', valid_macro_f1,  valid_acc, best_macro_f1, best_acc)
      writer.flush()
      writer.close()
      logging.info('name %s, best_acc %f, best_macro_f1 %f', name, best_acc, best_macro_f1)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  macro_f1 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.type(torch.FloatTensor)
    input = Variable(input).cuda()
    target = Variable(target).cuda(async=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target.long())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
    macro = utils.f1(logits, target)
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    macro_f1.update(macro, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f ', step, objs.avg, top1.avg, macro_f1.avg)

  return top1.avg, objs.avg, macro_f1.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  macro_f1 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.type(torch.FloatTensor)
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target.long())

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
    macro = utils.f1(logits, target)
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    macro_f1.update(macro, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f ', step, objs.avg, top1.avg, macro_f1.avg)

  return top1.avg, objs.avg, macro_f1.avg


if __name__ == '__main__':
  main() 

