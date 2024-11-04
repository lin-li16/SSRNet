import numpy as np
import torch
import torch.nn as nn
import warnings
import time
import sys
import scipy.io
from net_allsites import *
from solver import *
from eventDataset import *
import optuna
warnings.filterwarnings("ignore")


class Logger(object):
    '''
    log文件记录对象，将所有print信息记录在log文件中
    '''
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


def objective(trial):

    # 定义需要优化的超参数搜寻范围
    learning_rate = trial.suggest_float('lr', 1e-4, 1e-2,step=0.0001)
    ker1 = trial.suggest_int('ker1', 3, 31, step=2)
    ker2 = trial.suggest_int('ker2', 3, 31, step=2)
    nums = trial.suggest_int('nums', 32, 512)
    batch = trial.suggest_int('batch', 256, 2048)
    max_epoch = 500
    disp_freq = -1

    # 加载数据
    dataset = scipy.io.loadmat('all_sites/dataset_all2.mat')
    train_dataset = eqkDataset(dataset['train_data'], dataset['train_label'])
    valid_dataset = eqkDataset(dataset['valid_data'], dataset['valid_label'])
    test_dataset = eqkDataset(dataset['test_data'], dataset['test_label'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)
    valid_loader = torch.utils.data.DataLoader(valid_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch)
    Net = CNN_allsites(ker1=ker1, ker2=ker2, step=1, nums=nums)  
    # GPU加速
    if torch.cuda.is_available():
        Net = Net.cuda()
    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    slvr = Solver(Net, criterion, optimizer, train_loader, valid_loader)
    starttime = time.time()
    slvr.train(max_epoch, disp_freq, check_points=0)
    train_time = time.time()-starttime
    print('Training Time {:.4f}'.format(train_time))
    _, test_loss = test(slvr.valid_best_model, criterion, test_loader, batch=batch)
    torch.cuda.empty_cache()
    return test_loss


def main():
    sys.stdout = Logger('HPO_log.log')      # 创建log文件对象
    # 运行Optuna优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=25)

    # 获取最佳超参数
    best_params = study.best_params
    print("最佳超参数：", best_params)
    study.trials_dataframe().to_excel('HPO_optuna.xlsx')


if __name__ == "__main__":
    main()