import argparse
import torch
import os
from matplotlib import pyplot as plt
import numpy as np
import random
import joblib

from utils import rolling_window, acf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nz', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--seq_len', type=int, default=127)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')

    args = parser.parse_args()
    return args


def infer(args):
    nz = args.nz
    batch_size = args.batch_size
    seq_len = args.seq_len

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_processor = joblib.load(os.path.join(args.log_dir, 'data_processor.joblib'))
    generator = torch.load(os.path.join(args.log_dir, 'generator.pth')).to(device)
    generator.eval()
    
    noise = torch.randn(batch_size, nz, seq_len).to(device)
    with torch.no_grad():
        gen_y = generator(noise).cpu().detach().squeeze()
    
    y = data_processor.postprocess(gen_y)

    # Chart 1
    _, ax = plt.subplots(figsize=(16,9))
    ax.plot(np.cumsum(y[0:30], axis=1).T, alpha=0.75)
    ax.set_title('30 generated log return paths'.format(len(y)))
    ax.set_xlabel('days')
    ax.set_ylabel('Cumalative log return')
    plt.savefig(os.path.join(args.log_dir, 'cumalative_log_return.png'))

    log_returns = data_processor.log_returns
    # Chart 2
    n_bins = 50
    windows = [1, 5, 20, 100]
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    for i in range(len(windows)):
        row = min(max(0, i-1), 1)
        col = i % 2
        real_dist = rolling_window(log_returns, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
        fake_dist = rolling_window(y.T, windows[i], sparse = not (windows[i] == 1)).sum(axis=0).ravel()
        axs[row, col].hist(np.array([real_dist, fake_dist], dtype='object'), bins=n_bins, density=True)
        axs[row,col].set_xlim(*np.quantile(fake_dist, [0.001, .999]))

        axs[row,col].set_title('{} day return distribution'.format(windows[i]), size=16)
        axs[row,col].yaxis.grid(True, alpha=0.5)
        axs[row,col].set_xlabel('Cumalative log return')
        axs[row,col].set_ylabel('Frequency')
    axs[0,0].legend(['Historical returns', 'Synthetic returns'])
    plt.savefig(os.path.join(args.log_dir, 'real_vs_synthetic_dist.png'))


    #Chart 3
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))

    axs[0,0].plot(acf(log_returns, 100))
    axs[0,0].plot(acf(y.T, 100).mean(axis=1))
    axs[0,0].set_ylim(-0.1, 0.1)
    axs[0,0].set_title('Identity log returns')
    axs[0,1].plot(acf(log_returns**2, 100))
    axs[0,1].set_ylim(-0.05, 0.5)
    axs[0,1].plot(acf(y.T**2, 100).mean(axis=1))
    axs[0,1].set_title('Squared log returns')
    axs[1,0].plot(abs(acf(log_returns, 100, le=True)))
    axs[1,0].plot(abs(acf(y.T, 100, le=True).mean(axis=1)))
    axs[1,0].set_ylim(-0.05, 0.4)
    axs[1,0].set_title('Absolute')
    axs[1,1].plot(acf(log_returns, 100, le=True))
    axs[1,1].plot(acf(y.T, 100, le=True).mean(axis=1))
    axs[1,1].set_ylim(-0.2, 0.1)
    axs[1,1].set_title('Leverage effect')


    for ax in axs.flat: 
        ax.grid(True)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
    plt.setp(axs, xlabel='Lag (number of days)')
    plt.savefig(os.path.join(args.log_dir, 'real_vs_synthetic_lag.png'))

if __name__ == '__main__':
    args = parse_args()
    infer(args)