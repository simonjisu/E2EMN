# -*- coding utf-8 -*-
# author: simonjisu
# date: 18.11.07
import os
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pylab as plt
import matplotlib.ticker as ticker



class Config(object):
    def __init__(self, emd):
        self.ROOT = os.path.join(self.check_path('data'), 'QA_bAbI_tasks/en-valid-10k/')
        self.BATCH = 32
        self.FIX_STORY = 50
        self.LR = 0.01
        self.ANNEAL = 0.5
        self.EMBED = emd
    
    def check_path(self, folder):
        if os.path.isdir(os.path.join(os.getcwd(), folder)):
            path = os.path.relpath(os.path.join(os.getcwd(), folder))
        else:
            path = os.path.relpath(os.path.join(os.path.split(os.getcwd())[0], folder))
        return path
    
    def build(self, task, weight_style, other_method, ls=False):
        """
        return model load path
        """
        path = self.check_path('saved_models')
        self.TASK = task
        self.W_STYLE = weight_style
        self.ENC_METHOD = 'pe' if 'pe' in other_method.split('_') else 'bow'
        self.TEMPROAL = True if 'te' in other_method.split('_') else False
        self.LS = ls
        return os.path.join(path, weight_style, other_method, 'task{}.model'.format(task))

############ Functions ############
def get_story_idx(x, pad_idx):
    story_mask = x.eq(pad_idx).eq(pad_idx).sum(2).ge(1)  # B, len_story
    story_idx = torch.arange(1, story_mask.size(1)+1, device=x.device)
    story_idx = story_idx * story_mask.long()
    return torch.sort(story_idx, dim=1, descending=True)[0]


def matshow(array):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(array)
    fig.colorbar(cax)
    plt.show()
    
def plot_result(array, stories, questions, predict, answer, hops=3):
    """
    array: numpy array
    stories: list stories of tokens
    questions: questions of tokens
    predict, answer = string
    """
    sents = [' '.join(x) for x in stories]
    f, (ax_fig, ax_text) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[4, 1]}, dpi=100)
    cax = ax_fig.matshow(array, aspect="auto")
    f.colorbar(cax, ax=ax_fig)
    ax_fig.set_yticklabels([''] + sents)
    ax_fig.set_xticklabels([''] + ['hop {}'.format(h) for h in range(1, hops+1)])
    ax_fig.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax_fig.yaxis.set_major_locator(ticker.MultipleLocator(1))
    for (i, j), z in np.ndenumerate(array):
        ax_fig.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', 
                    fontdict={'color':'red', 'fontsize': 8})


    ax_text.axis('off')
    ax_text.text(-0, 0, 'Question: '+' '.join(questions)+'?', fontdict={'fontsize': 14})
    ax_text.text(-0, -0.5, 'Answer: '+ answer, fontdict={'fontsize': 14})
    ax_text.text(-0, -1, 'Predict: '+ predict, fontdict={'fontsize': 14})

    f.tight_layout()
    plt.show()
    plt.close()

    
def run_single_example(stories, questions, model, dataset):
    s, q, _ = map(torch.LongTensor, dataset._preprocess([stories], [questions], ['<unk>']))
    s_idx = get_story_idx(s, model.pad_idx)
    pred, ps = model(s, q, s_idx, return_p=True)
    attns = torch.stack([x.detach().masked_select(s_idx.ge(1)) for x in ps]).t().numpy()
    
    return pred.max(1)[1].item(), attns

def get_words(x, vocab, unk_idx=1):
    try:
        return vocab.itos[x]
    except:
        return vocab.itos[unk_idx]
    
def decode(idxes, vocab):
    if not isinstance(idxes, list):
        idxes = [idxes]
    return [get_words(x, vocab=vocab) for x in idxes]