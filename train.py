import argparse
import os
import torch
import torch.optim as optim
from tqdm.auto import tqdm
import joblib
import random
import numpy as np

from data import StockDataset, DataProcessor
from model import Generator, Discriminator
from utils import plot_loss

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--nz', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--seq_len', type=int, default=127)
    parser.add_argument('--clip', type=float, default=0.01)
    parser.add_argument('--lr', type=int, default=0.0002)
    parser.add_argument('--train_gen_per_epoch', type=int, default=5)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')
    args= parser.parse_args()
    return args
    

def train(args=None):
    data_processor = DataProcessor('Adj Close')
    log_returns_preprocessed = data_processor.preprocess(args.data_path)
    
    num_epochs = args.num_epochs
    nz = args.nz
    batch_size = args.batch_size
    seq_len = args.seq_len
    clip= args.clip
    lr = args.lr

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Setup the dataloader
    dataset = StockDataset(log_returns_preprocessed, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    progressing_bar = tqdm(range(num_epochs))

    history = dict(gen_loss=[], disc_loss=[])

    # Initialize the generator and discriminator
    generator = Generator().to(device)
    discriminator = Discriminator(seq_len).to(device)
    
    # Setup the optimizer
    disc_optimizer = optim.RMSprop(discriminator.parameters(), lr=lr)
    gen_optimizer = optim.RMSprop(generator.parameters(), lr=lr)

    # Training loop
    for epoch in progressing_bar:
        progressing_bar.set_description('Epoch %d' % (epoch))

        for idx, data in enumerate(dataloader, 0):
            
            # Train the discriminator
            discriminator.zero_grad()
            real = data.to(device)
            batch_size, seq_len = real.size(0), real.size(1)
            noise = torch.randn(batch_size, nz, seq_len, device=device)
            fake = generator(noise).detach()

            disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
            disc_loss.backward()
            disc_optimizer.step()

            for dp in discriminator.parameters():
                dp.data.clamp_(-clip, clip)

            # Train the generator
            if idx % args.train_gen_per_epoch == 0:
                generator.zero_grad()
                gen_loss = -torch.mean(discriminator(generator(noise)))
                gen_loss.backward()
                gen_optimizer.step()   

            history['gen_loss'].append(gen_loss.item())
            history['disc_loss'].append(disc_loss.item())
        progressing_bar.set_postfix_str('DiscLoss: %.4e, GenLoss: %.4e' % (disc_loss.item(), gen_loss.item()))
        
    plot_loss(history, os.path.join(args.log_dir, 'training_loss.png'))
    joblib.dump(data_processor, os.path.join(args.log_dir, 'data_processor.joblib'))
    torch.save(generator, os.path.join(args.log_dir, 'generator.pth'))

if __name__ == '__main__':
    args = parse_args()
    train(args)