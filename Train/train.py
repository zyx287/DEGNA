import os
from time import *
import argparse
import time
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from regression_model import EGNA
from multi_loader import *

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--protein", type=str, help="The PDB file of the protein")
    parser.add_argument("-l", "--ligand", type=str, help="The sdf or mol2 file of the ligand")
    parser.add_argument("-f", "--format", type=str, default="mol2", help='The format of the ligand file')
    parser.add_argument("-d", "--database", type=str, help="The path of the sequence database for hhblits")
    parser.add_argument("-u", "--cpu", type=int, default=4, help="The number of cpu cores used for hhblits")
    parser.add_argument("-o", "--output", type=str, default="", help="The path of the output file")
    parser.add_argument("-b", "--batch",type=int, default=256, help="Number of batch")
    parser.add_argument("-e", "--epoch",type=int, default=30, help="Training epochs")
    return parser.parse_args()

def train_parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, help='name of the model to be stored.')
    parser.add_argument('-t', '--train', help='train or predict the model', action='store_true')
    parser.add_argument('-d', '--dataset_name', type=str, help='name of dataset')
    parser.add_argument('-f', '--fold', type=int, help='current fold of cv')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size (only used for training)')
    parser.add_argument('-e', '--epoch_num', type=int, help='the number of epoch (only used for training)')
    parser.add_argument('-l', '--learning_rate', type=float, help='learning rate (only used for training)')
    return parser.parse_args()

class RegressionLoss(torch.nn.Module):
    def __init__(self) -> None:
        super(RegressionLoss,self).__init__()
    def forward(self, output, true_data):
        mse_loss = torch.mean((output - true_data)**2)
        mae_loss = torch.mean((torch.abs(output-true_data)))
        # Reweighting
        alpha = 0.5
        final_loss = alpha*mse_loss + (1-alpha)*mae_loss
        return final_loss

class TrainTest:
    def __init__(self, train, dataset_name, fold_k, tr_bz, lr, model_path, logger):
        begin_time = time()
        ###Loding Data ... ...###
        self.logger = logger

        if train:
            write_log(logger)
            self.device = torch.device("cuda", self.local_rank)

            self.tr_bz = tr_bz
            self.train_data = BindingData(tr_bz, logger, fold_k)

            self.train_dataloader = DataLoader(dataset=self.train_data, batch_size=self.tr_bz, 
                                               shuffle=False)
        self.net = EGNA()

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
        self.criterion = RegressionLoss()
        ## Load multi data 
        ## Shared Variables
        self.model_path = model_path
        self.n_fea_types = 4
        self.iter_num = 0

    def training(self, n_epoch, epoch_base):
        max_auc = 0
        for epoch in range(epoch_base, epoch_base + n_epoch):
            write_log(self.logger, "Training")
            train_loss = self.train_one_epoch(epoch)
    
    def train_one_epoch(self, epoch):
        begin_time = time()

        self.net.train()
        training_loss_sum = 0
        loss_history = torch.tensor([0, 0], dtype=torch.float16).to(self.device)
        n_iter_per_epoch = 500
        for i, data in enumerate(self.train_dataloader):
            for j in range(self.n_fea_types):
                data[j] = data[j].to(self.device)
            predictions, labels = self.net(data[:self.n_fea_types-2],False)
            self.optimizer.zero_grad()
            loss = self.criterion(predictions, labels)
            loss.backward()
            training_loss_sum += float(loss.data)
            self.optimizer.step()

            if i % n_iter_per_epoch == n_iter_per_epoch - 1:
                self.iter_num += n_iter_per_epoch
                write_log(self.logger, "Epoch:{}, iteration: {}, current loss: {}".format(
                    epoch, i + 1, training_loss_sum / n_iter_per_epoch))
                loss_history[0] += training_loss_sum / float(n_iter_per_epoch)
                loss_history[1] += 1.0
                training_loss_sum = 0

                end_time = time()
                training_time = (end_time - begin_time) / 60
                mean_loss = loss_history[0] / loss_history[1]
                write_log(self.logger, "Training time for epoch {}: {:.2F} min".format(epoch, training_time))
                return mean_loss


        
def main():
    args = train_parse_argument()
    logger = log_config('{}.log'.format(args.model_name))
    model_path = 'saved_model/{}.pkl'.format(args.model_name)
    try:
        assert args.batch_size == 0
        train_task = TrainTest(args.train, args.dataset_name, args.fold, args.batch_size,
                         args.learning_rate,
                         model_path, logger)
        train_task.testing(0, False)
        train_task.training(args.epoch_num, args.epoch_base)
    except:
        print('Train False')


if __name__ == '__main__':
    args = parse_args()
    checkargs(args)
    opt = Config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sys.stdout = Logger(opt.model_path + '/training.log')
    main(opt,device)
    sys.stdout.log.close()