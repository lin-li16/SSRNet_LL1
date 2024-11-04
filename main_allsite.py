# 导入必要的库
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import warnings
import time
import os
import sys
import scipy.io
import pickle
from tqdm import tqdm
from net_allsites import *
from solver import *
from eventDataset import *
from torchinfo import summary
from plot import plot_loss
warnings.filterwarnings("ignore")
sns.set_style('ticks')
sns.set_context("poster")
plt.rcParams['font.sans-serif'] = 'Arial'


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


def main():
    # 添加命令行输入参数
    parser = argparse.ArgumentParser(description='DNN Model for Time Series Forecasting in KiK-Net Downhole Array Dataset')
    parser.add_argument('--path', type=str, default='all_sites', help='Parent file path of the dataset and the results')
    parser.add_argument('--batch', type=int, default=1024, help='Batch size of training data')
    parser.add_argument('--validratio', type=float, default=0.1, help='Ratio of validation data in all data, 0-1.0')
    parser.add_argument('--testratio', type=float, default=0.2, help='Ratio of test data in all data, 0-1.0')
    parser.add_argument('--fixedorder', type=int, default=0, help='Whether to use the former data order')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--printfreq', type=int, default=-1, help='Training message print frequency in each epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model', type=str, default='Basic_3inp', help='Type of model used in this dataset')
    parser.add_argument('--resultspath', type=str, default='results', help='File path of results')
    parser.add_argument('--step', type=int, default=3, help='Number of layers in LSTM')
    parser.add_argument('--nums', type=int, default=256, help='Number of layers in LSTM')
    parser.add_argument('--ker1', type=int, default=5, help='Kernel size used in CNN layers')
    parser.add_argument('--ker2', type=int, default=11, help='Kernel size used in CNN layers')
    parser.add_argument('--normalize', type=str, default='minmax', help='Normalization of the dataset')
    parser.add_argument('--pretrain', type=str, default='no', help='Use the pre-trained model')
    parser.add_argument('--checkpoints', type=int, default=0, help='Output the models at all epochs')
    parser.add_argument('--plots', type=int, default=1, help='Plot the figures for each earthquake')
    parser.add_argument('--noisy', type=float, default=0, help='Noisy level added to the data')
    parser.add_argument('--bias', type=int, default=1, help='Bias in network')
    parser.add_argument('--datapre', type=str, default=None, help='Dataset file')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout ratio')
    args = parser.parse_args()

    # 创建结果文件夹
    results_path = os.path.join(args.path, args.resultspath)
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        # os.mkdir(os.path.join(results_path, 'figures'))
    sys.stdout = Logger(os.path.join(results_path, 'message.log'))      # 创建log文件对象
    print('The path of the results is %s' % results_path)

    if args.datapre is None:
        ## 导入数据
        inp0 = np.load('all_sites/inp0.npy').astype(np.float32)
        inp1 = np.load('all_sites/inp1.npy').astype(np.float32)
        inp = np.concatenate((inp0.reshape((inp0.shape[0], -1)), inp1.reshape((inp1.shape[0], -1))), axis=1)
        out = np.load('all_sites/out.npy').astype(np.float32)
        out = out.reshape((out.shape[0], -1))

        ## 构造训练集、验证集和测试集
        valid_size = args.validratio
        test_size = args.testratio
        train_size = 1 - valid_size - test_size

        ## 每个台站按比例取数据，且只取小震数据（烈度小于3）
        # idx_strong = np.where(inp[0][:, -1, 0] > 4)[0]
        # idx_small = np.where(inp0[:, -1, 0] <= 3)[0]
        # # np.random.shuffle(idx_small)
        # nums = len(idx_small)
        # train_idx = idx_small[: int(train_size * nums)]
        # valid_idx = idx_small[int(train_size * nums) : int((train_size + valid_size) * nums)]
        # test_idx = idx_small[int((train_size + valid_size) * nums) :]

        ### 每个台站数据按比例划分
        id_sites = np.load('all_sites/id_site.npy')
        train_idx, valid_idx, test_idx = [], [], []
        site_list = np.unique(np.array(id_sites))
        for site in site_list:
            idx_site = np.where(id_sites == site)[0]
            ima_sta = inp0[idx_site, -1, 0]
            idx_site_small = idx_site[ima_sta <= 3]
            # idx_site_strong = idx_site[ima_sta > 4]
            # np.random.shuffle(idx_site_small)
            # np.random.shuffle(idx_site_strong)
            num = len(idx_site_small)
            train_idx += list(idx_site_small[:int(train_size * num)])
            valid_idx += list(idx_site_small[int(train_size * num) : int((train_size + valid_size) * num)])
            test_idx += list(idx_site_small[int((train_size + valid_size) * num):])

        print('Train data size: %d, Valid data size: %d, test data size: %d, batch size: %d' % (len(train_idx), len(valid_idx), len(test_idx), args.batch))

        train_data, train_label = inp[train_idx, ...], out[train_idx, :]
        valid_data, valid_label = inp[valid_idx, ...], out[valid_idx, :]
        test_data, test_label = inp[test_idx, ...], out[test_idx, :]

        ## 数据标准化
        if args.normalize == 'minmax':
            data_min, data_max = np.min(train_data, axis=0), np.max(train_data, axis=0)
            label_min, label_max = np.min(train_label, axis=0), np.max(train_label, axis=0)
            train_data = (train_data - data_min) / (data_max - data_min)
            valid_data = (valid_data - data_min) / (data_max - data_min)
            test_data = (test_data - data_min) / (data_max - data_min)
            train_label = (train_label - label_min) / (label_max - label_min)
            valid_label = (valid_label - label_min) / (label_max - label_min)
            test_label = (test_label - label_min) / (label_max - label_min)
            scipy.io.savemat(os.path.join(results_path, 'dataset.mat'),
                        {'train_data': train_data, 'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label, 'valid_data': valid_data, 'valid_label': valid_label, 'data_min': data_min, 'data_max': data_max, 'label_min': label_min, 'label_max': label_max})
        elif args.normalize == 'standard':
            data_mean, data_std = np.mean(train_data, axis=0), np.std(train_data, axis=0)
            label_mean, label_std = np.mean(train_label, axis=0), np.std(train_label, axis=0)
            train_data = (train_data - data_mean) / data_std
            valid_data = (valid_data - data_mean) / data_std
            test_data = (test_data - data_mean) / data_std
            train_label = (train_label - label_mean) / label_std
            valid_label = (valid_label - label_mean) / label_std
            test_label = (test_label - label_mean) / label_std
            scipy.io.savemat(os.path.join(results_path, 'dataset.mat'),
                        {'train_data': train_data, 'train_idx': train_idx, 'valid_idx': valid_idx, 'test_idx': test_idx, 'train_label': train_label, 'test_data': test_data, 'test_label': test_label, 'valid_data': valid_data, 'valid_label': valid_label, 'data_mean': data_mean, 'data_std': data_std, 'label_mean': label_mean, 'label_std': label_std})

        train_dataset = eqkDataset(train_data, train_label)
        valid_dataset = eqkDataset(valid_data, valid_label)
        test_dataset = eqkDataset(test_data, test_label)
    else:
        dataset = scipy.io.loadmat(os.path.join(args.path, args.datapre))
        train_dataset = eqkDataset(dataset['train_data'], dataset['train_label'])
        valid_dataset = eqkDataset(dataset['valid_data'], dataset['valid_label'])
        test_dataset = eqkDataset(dataset['test_data'], dataset['test_label'])
        train_data, train_label = dataset['train_data'], dataset['train_label']
        valid_data, valid_label = dataset['valid_data'], dataset['valid_label']
        test_data, test_label = dataset['test_data'], dataset['test_label']
        data_min, data_max = dataset['data_min'], dataset['data_max']
        label_min, label_max = dataset['label_min'], dataset['label_max']

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch)
    valid_loader = torch.utils.data.DataLoader(valid_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    
    # 进行训练
    max_epoch = args.epochs
    disp_freq = args.printfreq
    learning_rate = args.lr
    print('%s model is applied' % args.model)
    print('Learning rate is %f' % learning_rate)
    Net = CNN_allsites(ker1=args.ker1, ker2=args.ker2, step=args.step, nums=args.nums)  
    summary(Net)
    # GPU加速
    if torch.cuda.is_available():
        Net = Net.cuda()
    # optimizer = torch.optim.LBFGS(Net.parameters(), lr=learning_rate, max_iter=2)
    optimizer = torch.optim.Adam(Net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    slvr = Solver(Net, criterion, optimizer, train_loader, valid_loader)
    starttime = time.time()
    slvr.train(max_epoch, disp_freq, check_points=args.checkpoints)
    train_time = time.time()-starttime
    print('Training Time {:.4f}'.format(train_time))
    _, test_loss = test(slvr.valid_best_model, criterion, test_loader, batch=args.batch)
    print("Test Loss {:.4f}\n".format(test_loss))
    torch.cuda.empty_cache()

    # 绘制loss变化曲线
    plot_loss(slvr.avg_train_loss_set, slvr.avg_val_loss_set, yscale='log')
    plt.savefig(os.path.join(results_path, 'loss.svg'), bbox_inches='tight')
    print('Training best epoch: %d\tTraining minimum loss: %.3E' % (np.argmin(slvr.avg_train_loss_set) + 1, np.min(slvr.avg_train_loss_set)))
    print('Validate best epoch: %d\tValidate minimum loss: %.3E' % (np.argmin(slvr.avg_val_loss_set) + 1, np.min(slvr.avg_val_loss_set)))
    torch.save(slvr.train_best_model, os.path.join(results_path, 'trainbest.pt'))
    torch.save(slvr.valid_best_model, os.path.join(results_path, 'validbest.pt'))
    if args.checkpoints==0:
        torch.save(slvr.net, os.path.join(results_path, 'last.pt'))
    else:
        torch.save(slvr.all_models, os.path.join(results_path, 'allmodels.pt'))  

    # 结果评价
    train_pred, _ = test(slvr.valid_best_model, criterion, torch.utils.data.DataLoader(train_dataset), batch=args.batch)
    valid_pred, _ = test(slvr.valid_best_model, criterion, valid_loader, batch=args.batch)
    test_pred, _ = test(slvr.valid_best_model, criterion, test_loader, batch=args.batch) 
    if args.normalize == 'minmax':
        train_pred = train_pred * (label_max - label_min) + label_min
        valid_pred = valid_pred * (label_max - label_min) + label_min
        test_pred = test_pred * (label_max - label_min) + label_min
        train_label = train_label * (label_max - label_min) + label_min
        valid_label = valid_label * (label_max - label_min) + label_min
        test_label = test_label * (label_max - label_min) + label_min
    elif args.normalize == 'standard':
        train_pred = train_pred * label_std + label_mean
        valid_pred = valid_pred * label_std + label_mean
        test_pred = test_pred * label_std + label_mean
        train_label = train_label * label_std + label_mean
        valid_label = valid_label * label_std + label_mean
        test_label = test_label * label_std + label_mean

    scipy.io.savemat(os.path.join(results_path, 'results.mat'),
                     {'train_label': train_label, 'train_pred': train_pred,
                      'valid_label': valid_label, 'valid_pred': valid_pred,
                      'test_label': test_label, 'test_pred': test_pred})

    with open(os.path.join(results_path, 'performance.out'), 'w') as out_file:
        out_file.write('训练最好次数：%d\n' % (np.argmin(slvr.avg_train_loss_set) + 1))
        out_file.write('验证最好次数：%d\n' % (np.argmin(slvr.avg_val_loss_set) + 1))
        out_file.write('%10s:%10.3E\n' % ('train-RMSE', np.sqrt(np.mean((train_pred - train_label) ** 2))))
        out_file.write('%10s:%10.3E\n' % ('train-MAE', np.mean(np.abs(train_pred - train_label))))
        out_file.write('%10s:%10.3f%%\n' % ('train-r', 100 * np.corrcoef(train_pred.ravel(), train_label.ravel())[0, 1]))
        out_file.write('%10s:%10.3E\n' % ('valid-RMSE', np.sqrt(np.mean((valid_pred - valid_label) ** 2))))
        out_file.write('%10s:%10.3E\n' % ('valid-MAE', np.mean(np.abs(valid_pred - valid_label))))
        out_file.write('%10s:%10.3f%%\n' % ('valid-r', 100 * np.corrcoef(valid_pred.ravel(), valid_label.ravel())[0, 1]))
        out_file.write('%10s:%10.3E\n' % ('test-RMSE', np.sqrt(np.mean((test_pred - test_label) ** 2))))
        out_file.write('%10s:%10.3E\n' % ('test-MAE', np.mean(np.abs(test_pred - test_label))))
        out_file.write('%10s:%10.3f%%\n' % ('test-r', 100 * np.corrcoef(test_pred.ravel(), test_label.ravel())[0, 1]))


if __name__ == "__main__":
    main()