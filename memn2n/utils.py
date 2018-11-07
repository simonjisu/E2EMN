# -*- coding utf-8 -*-
# author: simonjisu
# date: 18.11.07
import torch
import matplotlib.pylab as plt

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