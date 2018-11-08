# -*- coding utf-8 -*-
# author: simonjisu
# date: 18.11.08

# import packages
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model import MEMN2N
from bAbIDataSet_utils import bAbI
from utils import get_story_idx

def import_data(config, device, is_test=False):
    babi = bAbI()
    train, valid, test = babi.splits(root=config.ROOT, 
                                     task=config.TASK, 
                                     fix_maxlen_story=config.FIX_STORY,
                                     device=device)
    train_loader, valid_loader, test_loader = babi.iters(train, valid, test, config.BATCH)
    if is_test:
        return train, test, test_loader
    else:
        return train, train_loader, valid_loader
    
def build_model(config, vocab, maxlen, device):
    model = MEMN2N(vocab_size=len(vocab), 
                   embed_size=config.EMBED, 
                   weight_style=config.W_STYLE,
                   encoding_method=config.ENC_METHOD,
                   temporal=config.TEMPROAL, 
                   maxlen_story=maxlen,
                   pad_idx=vocab.stoi['<pad>']).to(device)

    loss_function = nn.NLLLoss(ignore_index=vocab.stoi['<pad>'], reduction='sum')
    optimizer = optim.SGD(model.parameters(), lr=config.LR)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=config.ANNEAL, milestones=[25, 50, 75], 
                                               optimizer=optimizer)
    
    return model, loss_function, optimizer, scheduler

def run_step(config, vocab, loader, model, loss_function, optimizer, ls=False):
    model.train()
    losses=[]
    for batch in loader:
        stories, queries, answers = batch
        stories_idx = get_story_idx(stories, vocab.stoi['<pad>'])
        model.zero_grad()
        
        nll_loss = model(stories, queries, stories_idx, ls=ls)
        loss = loss_function(nll_loss, answers.view(-1))
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40.0)  # gradient clipping
        optimizer.step()
        
    return np.mean(losses)

def validation(config, vocab, loader, model, loss_function, ls=False):
    model.eval()
    losses=[]
    for batch in loader:
        stories, queries, answers = batch
        stories_idx = get_story_idx(stories, vocab.stoi['<pad>'])        
        nll_loss = model(stories, queries, stories_idx, ls=ls)
        loss = loss_function(nll_loss, answers.view(-1))
        losses.append(loss.item())
        
    return np.mean(losses)
        
def train_model(config, model, vocab, loss_function, optimizer, scheduler, train_loader, valid_loader):
    valid_losses = [9999, 9999]
    wait = 0
    ls = True if config.LS else False
    print('--'*20)
    start_time = time.time()
    for step in range(config.STEP):
        scheduler.step()
        train_loss = run_step(config, vocab, train_loader, model, loss_function, optimizer, ls=ls)
        valid_loss = validation(config, vocab, valid_loader, model, loss_function, ls=ls)
        valid_losses.append(valid_loss)
        
        if config.EMPTY_CUDA_MEMORY:
            torch.cuda.empty_cache()
        
        if step % config.PRINT_EVERY == 0:
            print('[{}/{}] (train) loss {:.4f} | (valid) loss {:.4f}'.format(
                        step+1, config.STEP, train_loss, valid_loss))
        # Turn off Linear Start, when valid loss start to go up
        if config.LS & (valid_losses[-1] > valid_losses[-2]):
            ls = False

        # Save model
        if config.SAVE_MODEL:
            if config.SAVE_BEST:
                if valid_loss <= min(valid_losses):
                    torch.save(model.state_dict(), config.SAVE_PATH)
            else:
                model_path = config.SAVE_PATH + \
                    '{}_{:.4f}_{:.4f}'.format(step, train_loss, valid_loss)
                torch.save(model.state_dict(), model_path)
                
        # EarlyStopping
        if valid_loss > min(valid_losses):
            wait += 1
            if wait > config.THRES:
                print('EARLY BREAK: valid loss goes over the patient number {}'.format(config.THRES))
                break
        elif valid_loss == 0.0:
            print('EARLY BREAK: valid loss equals to zero')
            break
            
    end_time = time.time()
    total_time = end_time-start_time
    hour = int(total_time // (60*60))
    minute = int((total_time - hour*60*60) // 60)
    second = total_time - hour*60*60 - minute*60
    print('\nTraining Excution time with validation: {:d} h {:d} m {:.4f} s'.format(hour, minute, second))
        
    
    
    

