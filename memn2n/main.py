# -*- coding utf-8 -*-
# author: simonjisu
# date: 18.11.08

import os
import argparse
import torch
from train import import_data, build_model, train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MEMN2N argument parser')
    # loader
    parser.add_argument('-root', '--ROOT', help='Location of path', type=str, 
                        default='./data/QA_bAbI_tasks/en-valid-10k/')
    parser.add_argument('-task', '--TASK', help='Training task number from 1~20, if "0" do all task',
                        type=int, default=1)
    parser.add_argument('-fixlen', '--FIX_STORY', help='If "0" dont fix story lengths', type=int, default=50)
    parser.add_argument('-bs', '--BATCH', help='Batch Size', type=int, default=32)
    parser.add_argument('-cuda', '--USE_CUDA', help='Use cuda if exists', action='store_true')
    parser.add_argument('-emptymem', '--EMPTY_CUDA_MEMORY', help='Use cuda empty cashce', action='store_true')
    
    # model
    parser.add_argument('-emd', '--EMBED', help='Embedding size', type=int, default=20)
    parser.add_argument('-wstyle', '--W_STYLE', help='Weight Sharing Style, "adjacent" or "rnnlike"', type=str, default='adjacent')
    parser.add_argument('-encmth', '--ENC_METHOD', help='Encoding Method, "bow" or "pe"', type=str, default='bow')
    parser.add_argument('-temporal', '--TEMPROAL', help='Whether to use Temporal Encoding', action='store_true')
    parser.add_argument('-ls', '--LS', help='Whether to use Linear Start', action='store_true')

    # optimizer
    parser.add_argument('-lr', '--LR', help='Learning rate', type=float, default=0.01)
    parser.add_argument('-stp', '--STEP', help='Learning Steps', type=int, default=100)
    parser.add_argument('-anl', '--ANNEAL', help='Learning rate anneal', type=float, default=0.5)

    # save model
    parser.add_argument('-save', '--SAVE_MODEL', help='Save model', action='store_true')
    parser.add_argument('-savebest', '--SAVE_BEST', help='Save best model', action='store_true')
    parser.add_argument('-svp', '--SAVE_PATH', help='Saving model path', type=str, default='./saved_models/memn2n.model')
    # others 
    parser.add_argument('-pes', '--PRINT_EVERY', help='Print every step ', type=int, default=1)
    parser.add_argument('-thres', '--THRES', help='Earlystopping patience number', type=int, default=5)

    config = parser.parse_args()
    
    if config.USE_CUDA:
        assert config.USE_CUDA == torch.cuda.is_available(), 'cuda is not avaliable.'
    DEVICE = 'cuda' if config.USE_CUDA else None
    
    if config.TASK != 0:
        if config.FIX_STORY == 0:
            config.FIX_STORY = None
        print(config)
        train, train_loader, valid_loader = import_data(config, DEVICE, is_test=False)
        model, loss_function, optimizer, scheduler = build_model(config, train.vocab, train.maxlen_story, DEVICE)
        train_model(config, model, train.vocab, loss_function, optimizer, scheduler, train_loader, valid_loader)
# [vocab issue] for training jointly, share all vocab in the training. have to modify bAbIDataSet_uitls.py
#     else:
#         for i in range(1, 21):
#             config.TASK = i
#             print(config)
#             train, train_loader, valid_loader = import_data(config, DEVICE, is_test=False)
#             model, loss_function, optimizer, scheduler = build_model(config, train.vocab, train.maxlen_story, DEVICE)
#             train_model(config, model, train.vocab, loss_function, optimizer, scheduler, train_loader, valid_loader)
