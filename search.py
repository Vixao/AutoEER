import os
from random import shuffle
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
#from tensorboardX import SummaryReader
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold

from torch.autograd import Variable
from model_search import Network
from architect import Architect
from sklearn.model_selection import train_test_split
from utils import Temp_Scheduler
from genotypes import PRIMITIVES
#parser = argparse.ArgumentParser("cifar")
parser = argparse.ArgumentParser("deap")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')#SGD 0.1
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')#SGD 0.01
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=7, help='gpu device id')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')#SEED为62，DEAP为32
parser.add_argument('--layers', type=int, default=5, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--flag', type=str, default='v', help='valence or arousal or seed')
parser.add_argument('--temp', type=float, default=5.0, help='initial softmax temperature')
parser.add_argument('--temp_min', type=float, default=0.001, help='minimal softmax temperature')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
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
  
  subject_name = ['01','02', '03', '04', '05', '06', '07', '08', '09',
              '10', '11', '12', '13', '14', '15', '16', '17', '18',
              '19', '20', '21', '22', '23', '24', '25','26','27','28','29','30','31','32']

  for i in range(len(subject_name)):
      logging.info("processing: %s ......", subject_name[i])
      name=subject_name[i]
      #加载数据
      #DEAP
      dataset_dir = "/home/wyx/data/DEAP_DE/3D_2400_32_4_4/"
      print("loading ",dataset_dir+"DE_s"+name,".mat")
      data_file = sio.loadmat(dataset_dir+"DE_s"+name+".mat")
      data= data_file["data"].astype(np.float32)
      #data=data.transpose([0,2,1,3])
      print("operations",PRIMITIVES)
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
      train_data, val_data, train_label, val_label = train_test_split(datas["train"],labels["train"], 
                                      test_size=0.5, 
                                      stratify=labels["train"],
                                      shuffle=True,
                                      random_state=12)
      print("train_data shape:",train_data.shape)
      print("val_data shape:",val_data.shape)
      criterion = nn.CrossEntropyLoss()
      criterion = criterion.cuda()
      model = Network(args.init_channels, label_CLASSES, args.layers, criterion, args.temp)
      model = model.cuda()
      logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
      
      optimizer = torch.optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
      '''
      optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
      '''
      torch_data_train = GetLoader(train_data, train_label)
      torch_data_valid = GetLoader(val_data, val_label)

      train_queue = torch.utils.data.DataLoader(
      torch_data_train, batch_size=args.batch_size,
      pin_memory=True, num_workers=0)

      valid_queue = torch.utils.data.DataLoader(
      torch_data_valid, batch_size=args.batch_size,
      pin_memory=True, num_workers=0)

      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

      architect = Architect(model, args)
      temp_scheduler = Temp_Scheduler(args.epochs, model._temp, args.temp, temp_min=args.temp_min)
      best_acc=0
      log_dir = "logs/search/deap/"+args.flag+"/"+name
      if not os.path.exists(log_dir):
          os.makedirs(log_dir)
      writer = SummaryWriter(log_dir)
      for epoch in range(args.epochs):
        t1 = time.time()
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        model._temp = temp_scheduler.step()
        print(f'temperature: {model._temp}')
        lr = args.learning_rate
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
      # training
        train_acc, train_obj, train_f1 = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
        logging.info('train_acc %f, train_f1 %f', train_acc,train_f1)
        writer.add_scalar('Train/Loss', train_obj, epoch )
        writer.add_scalar('Train/Accuracy', train_acc, epoch )
        writer.add_scalar('Train/F1', train_f1, epoch )
      # validation
        valid_acc, valid_obj, valid_f1 = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f, valid_f1 %f', valid_acc,valid_f1)
        writer.add_scalar('Valid/Loss', valid_obj, epoch )
        writer.add_scalar('Valid/Accuracy', valid_acc, epoch )
        writer.add_scalar('Valid/F1', valid_f1, epoch )
        if valid_acc >= best_acc:
          best_genotype = genotype
          best_acc =valid_acc
        if args.epochs-epoch<=1:
          with open(args.flag+"_arch"+".txt","a+",encoding='utf-8') as f:
                f.write("{'"+name+"':"+'"'+str(best_genotype)+'"'+"}"+'\n')
          print("best_valid_acc",best_acc)
        path = os.path.join(args.save, name)
        if not os.path.exists(path):
          os.makedirs(path)
        utils.save(model, os.path.join(path,'weights.pt'))
        print(f'search time: {time.time() - t1}')
      writer.flush()
      writer.close()


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  f1 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda()

    if epoch>=8:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target.long())

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
    f1_score = utils.f1(logits, target)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    f1.update(f1_score, n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, f1.avg)

  return top1.avg, objs.avg, f1.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  f1 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = input.cuda()
      target = target.cuda()
    logits = model(input)
    loss = criterion(logits, target.long())

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 1))
    f1_score = utils.f1(logits, target)
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)
    f1.update(f1_score, n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, f1.avg)

  return top1.avg, objs.avg, f1.avg


if __name__ == '__main__':
  main() 


