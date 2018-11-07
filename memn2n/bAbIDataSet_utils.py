# -*- coding utf-8 -*-
# author: simonjisu
# date: 18.11.04

import os
from copy import deepcopy
from vocabulary import Vocab
import torch
import torch.utils.data as torchdata


class bAbI(object):
    def __init__(self):
        """
        root: please use en-valid-10k
        """
    def check_maxlen(self, x, maxlen):
        if len(x) >= maxlen:
            maxlen = len(x)
        return maxlen
    
    def load_data(self, path, fix_maxlen_story=None):
        maxlen_story = 0
        maxlen_query = 0
        maxlen_sent = 0
        all_datas = []
        story = []
        support_f = lambda x: [int(s) for s in x.split()]
        answer_f = lambda x: [a for a in x.split(',')]
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read().splitlines()
            if self.lower:
                data = [x.lower() for x in data]
            for line in data:
                idx, sentence = line.split(' ', 1)
                if int(idx) == 1:
                    maxlen_story = self.check_maxlen(story, maxlen_story)
                    story = []
                # query, answer, support
                if '?' in sentence:
                    q, a, s = sentence.split('\t')
                    q = q.strip().replace('?', '').split()
                    a = answer_f(a)
                    s = support_f(s)
                    maxlen_query = self.check_maxlen(q, maxlen_query)
                    # sentences are reversed order
                    temp = deepcopy(story)
                    temp = list(zip(*sorted(enumerate(temp), key=lambda x: x[0], reverse=True)))[1]
                    all_datas.append([temp, q, a, s])
                else:
                    sent = sentence.replace('.', '').split()
                    if fix_maxlen_story is not None:
                        if len(story) == fix_maxlen_story:
                            continue
                        else:
                            story.append(sent)
                    else:
                        story.append(sent)
                    maxlen_sent = self.check_maxlen(sent, maxlen_sent)
        datas = {'datas': all_datas, 
                 'maxlen_story': maxlen_story if fix_maxlen_story is None else fix_maxlen_story,
                 'maxlen_query': maxlen_query,
                 'maxlen_sent': maxlen_sent}    
        return datas
    
    def splits(self, root='../data/QA_bAbI_tasks/en-valid-10k/', task=1, fix_maxlen_story=None, lower=False, support=False, device=None):
        """
        root: dataset path, only can split train/valid/test, default is '../data/QA_bAbI_tasks/en-valid-10k/'
        task: task number
        """
        self.paths = [os.path.join(root, 'qa'+str(task)+'_'+x+'.txt') for x in ['train', 'valid', 'test']]
        self.lower = lower
        train_dict, valid_dict, test_dict = [self.load_data(path, fix_maxlen_story) for path in self.paths]
        train = bAbIDataset(train_dict, train=True, support=support, device=device)
        valid = bAbIDataset(valid_dict, train=False, vocabulary=train.vocab, support=support, device=device)
        test = bAbIDataset(test_dict, train=False, vocabulary=train.vocab, support=support, device=device)
        return train, valid, test
    
    def iters(self, train, valid, test, batch_size, shuffle=False):
        train_loader = torchdata.DataLoader(train, batch_size=batch_size, shuffle=shuffle, 
                                            collate_fn=train.collate_fn)
        valid_loader = torchdata.DataLoader(train, batch_size=batch_size, shuffle=shuffle, 
                                            collate_fn=valid.collate_fn)
        test_loader = torchdata.DataLoader(train, batch_size=batch_size, shuffle=shuffle, 
                                            collate_fn=test.collate_fn)
        return train_loader, valid_loader, test_loader
    
# bAbI Dataset 
class bAbIDataset(torchdata.Dataset):
    def __init__(self, data_dict, train=True, vocabulary=None, support=False, device=None):
        """
        'datas': all_datas
        'maxlen_story': maxlen_story
        'maxlen_query': maxlen_query
        'maxlen_sent': maxlen_sent
        """
        
        self.examples = data_dict['datas']
        self.maxlen_story = data_dict['maxlen_story']
        self.maxlen_query = data_dict['maxlen_query']
        self.maxlen_sent = data_dict['maxlen_sent']
        self.support = support
        self.device=device
        self.flatten = lambda x: [tkn for sublists in x for tkn in sublists]
        
        stories, questions, answers, supports = list(zip(*self.examples))
        if train:
            self.vocab = Vocab()
            self._build_vocab(stories, questions, answers)
        else:
            self.vocab = vocabulary
        # numerical & add_pad
        stories = [self.numerical(story) for story in stories]
        stories = [self.pad_sent(story, self.vocab.stoi['<pad>'], self.maxlen_sent) for story in stories]
        stories = self.pad_story(stories, self.vocab.stoi['<pad>'], self.maxlen_story, self.maxlen_sent)
        questions = self.numerical(questions)
        questions = self.pad_sent(questions, self.vocab.stoi['<pad>'], self.maxlen_query)
        answers = self.numerical(answers)
        
        if self.support:
            self.data = list(zip(stories, questions, answers, supports))
        else:
            self.data = list(zip(stories, questions, answers))
            
    def _build_vocab(self, stories, questions, answers):
        total_words = set(self.flatten(self.flatten(stories)) + self.flatten(questions) + self.flatten(answers))
        self.vocab.build_vocab(sorted(list(total_words)))
    
    def numerical(self, all_sents):
        # Numericalize all tokens
        f = lambda x: self.vocab.stoi.get(x) if self.vocab.stoi.get(x) is not None else self.vocab.stoi['<unk>']
        tokens_numerical = [list(map(f, sent)) for sent in all_sents]
        return tokens_numerical            
    
    def pad_sent(self, tokens_numerical, pad_idx, maxlen):
        return [sent+[pad_idx]*(maxlen-len(sent)) if len(sent) < maxlen else sent for sent in tokens_numerical]
    
    def pad_story(self, stories, pad_idx, maxlen_story, maxlen_sent):
        return [story + [[pad_idx]*maxlen_sent]*(maxlen_story-len(story)) \
                if len(story) < maxlen_story else story for story in stories]
    
    def collate_fn(self, data):
        """
        0: stories
        1: queries
        2: answers
        3: supports(optional)
        """
        f = lambda x: torch.LongTensor(x).to(self.device)
        return list(map(f, zip(*data)))

    def __getitem__(self, index):
        # return index datas
        return self.data[index]

    def __len__(self):
        # lengths of data
        return len(self.data)        
