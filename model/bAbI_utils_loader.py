# -*- coding utf-8 -*-
# author: simonjisu
# date: 18.04.11

import torch
from copy import deepcopy
import numpy as np
import random
flatten = lambda l: [item for sublist in l for item in sublist]


class bAbIDataset(object):
    def __init__(self, path, train=True, vocab=None, sos=None, eos=None, return_masks=False):
        """
        example:

        train(True): if you are in testing, please insert False
        vocab(None): if you are in testing, please insert [self.word2idx], self = training vocab in bAbIDataset class
        sos(None): start of sentence token
        eos(None): end of sentence token
        return_masks(False): if True, returns masked of batches
        """
        self.train = train
        self.return_masks = return_masks
        self.max_story_len = 0

        if self.train:
            data, vocab = self.bAbI_data_loader(path, vocab=None, sos=sos, eos=eos)
        else:
            assert vocab is not None, 'insert vocab = self.word2idx'
            data, vocab = self.bAbI_data_loader(path, vocab=vocab, sos=sos, eos=eos)

        self.word2idx = vocab
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.data = data

    def __len__(self):
        return len(self.data)

    def bAbI_data_loader(self, path, vocab=None, sos=None, eos=None):
        assert ((isinstance(sos, str) and isinstance(eos, str)) == True) or ((sos and eos) is None), print('error')
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = file.readlines()
            data = [l.strip() for l in data]
        except:
            print('no such file: {}'.format(path))
            return None

        data_temp = []
        story = []


        try:
            for line in data:
                idx, line = line.split(' ', 1)
                if idx == '1':
                    story = []

                if '?' in line:
                    q, a, support = line.split('\t')
                    q = q.lower().strip().replace('?', '').split() + ['?']
                    a = a.lower().strip().split() + [eos] if eos else a.lower().strip().split()
                    support = int(support)
                    story_temp = deepcopy(story)
                    data_temp.append([story_temp, q, a, support])
                else:
                    sentence = line.lower().replace('.', '').split() + [eos] if eos else line.lower().replace('.',
                                                                                                              '').split()
                    story.append(sentence)
        except:
            print('check data')
            return None

        if vocab:
            data, vocab = self.bAbI_build_vocab(data_temp, vocab)
        else:
            data, vocab = self.bAbI_build_vocab(data_temp, sos=sos, eos=eos)

        return data, vocab

    def bAbI_build_vocab(self, data, vocab=None, **kwargs):
        if vocab is None:
            sos = kwargs['sos']
            eos = kwargs['eos']
            story, q, a, s = list(zip(*data))
            vocab = list(set(flatten(flatten(story)) + flatten(q) + flatten(a)))
            if sos and eos:
                word2idx = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
            else:
                word2idx = {'<pad>': 0, '<unk>': 1}

            for word in vocab:
                if word2idx.get(word) is None:
                    word2idx[word] = len(word2idx)
        else:
            word2idx = vocab

        self.max_story_len = max(set([len(s) for s in list(zip(*data))[0]]))

        for d in data:
            # d[0]: stories
            # d[1]: questions
            # d[2]: answer
            # d[3]: support

            for i, story in enumerate(d[0]):
                d[0][i] = self._transfer2idx(story, word2idx)

            d[1] = self._transfer2idx(d[1], word2idx)
            d[2] = self._transfer2idx(d[2], word2idx)

        return data, word2idx

    def _transfer2idx(self, seq, dictionary):
        idxs = list(map(lambda w: dictionary[w] if dictionary.get(w) is not None else dictionary["<unk>"], seq))
        return idxs

    def pad_to_batch(self, batch, w2idx, no_batch=False):
        """
        stories, stories_masks: B, n, T_c
        questions, questions_masks: B, T_q
        answers: B, T_a
        supports: B
        -------------------------------------
        if return_masks == True:
            return: stories, stories_masks, questions, questions_masks, answers, supports
        else:
            return: stories, questions, answers, supports
        """
        story, q, a, s = list(zip(*batch))
        max_story = max([len(s) for s in story])  # max_stories
        max_len = max([len(s) for s in flatten(story)])  # max_sentence_len
        max_q = max([len(q_) for q_ in q])
        max_a = max([len(a_) for a_ in a])

        stories, stories_masks = [], []
        for i in range(len(batch)):
            story_array, story_mask = self.get_batch_array(self.get_fixed_array(story[i], w2idx), 
                                                           no_batch, max_story, max_len)
            stories.append(story_array)
            stories_masks.append(story_mask)

        questions, questions_masks = self.get_batch_array(self.get_fixed_array(q, w2idx), no_batch, len(batch), max_q)
        answers, _ = self.get_batch_array(self.get_fixed_array(a, w2idx), no_batch, len(batch), max_a)

        if self.return_masks:
            return stories, stories_masks, questions, questions_masks, answers, s
        else:
            return stories, questions, answers, s

    def get_fixed_array(self, data, w2idx):
        max_col = max([len(d) for d in data])
        for j in range(len(data)):
            if len(data[j]) < max_col:
                data[j].append(w2idx.get('<pad>'))
        return data

    def get_batch_array(self, data, no_batch, *shape):
        assert type(no_batch) == bool, 'no_batch, must be boolean'
        r, c = shape
        if no_batch:
            r = len(data)
        temp = np.zeros((r, c), dtype=np.int)
        it = np.nditer(np.array(data, dtype=np.int), flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = np.array(data)[idx]
            temp[idx] = tmp_val
            it.iternext()

        mask = (temp == 0).astype(np.byte)
        return temp.tolist(), mask.tolist()

    def pad_to_story(self, test_data, no_batch=True):  # this is for inference
        """
        if return_masks == True:
            return: stories(list), stories_masks(list), questions, questions_masks, answers, supports
        else:
            return: stories(list), questions, answers, supports
        """
        story, q, a, s = list(zip(*test_data))
        max_story = max([len(s) for s in story])  # max_stories
        max_len = max([len(s) for s in flatten(story)])  # max_sentence_len
        max_q = max([len(q_) for q_ in q])
        max_a = max([len(a_) for a_ in a])

        stories, stories_masks = [], []
        for i in range(len(test_data)):
            story_array, story_mask = self.get_batch_array(self.get_fixed_array(story[i], self.word2idx), no_batch, max_story,
                                                           max_len)
            stories.append(story_array)
            stories_masks.append(story_mask)

        questions, questions_masks = self.get_batch_array(self.get_fixed_array(q, self.word2idx), no_batch, len(test_data),
                                                          max_q)
        answers, _ = self.get_batch_array(self.get_fixed_array(a, self.word2idx), no_batch, len(test_data), max_a)

        if self.return_masks:
            return stories, stories_masks, questions, questions_masks, answers, s
        else:
            return stories, questions, answers, s


class bAbIDataLoader(object):
    def __init__(self, dataset, batch_size, shuffle=False, to_tensor=True):
        self.dataset = dataset
        self.train = dataset.train
        self.data = dataset.data
        self.return_masks = dataset.return_masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_tensor = to_tensor
        self.word2idx = dataset.word2idx
        self.idx2word = dataset.idx2word

    def _to_tensor(self, x):
        return torch.LongTensor(x)

    def load(self):
        if self.shuffle: random.shuffle(self.data)
        sindex = 0
        eindex = self.batch_size
        while eindex < len(self.data):
            batch = self.data[sindex: eindex]
            temp = eindex
            eindex = eindex + self.batch_size
            sindex = temp
            if self.to_tensor:
                if self.train:
                    batchs = self.dataset.pad_to_batch(batch, self.word2idx)
                    yield [self._to_tensor(x) for x in batchs]
                else:

                    batchs = self.dataset.pad_to_story(batch)
                    if self.return_masks:
                        yield [[self._to_tensor(y) for y in x] for x in batchs[:2]] + \
                                [self._to_tensor(x) for x in batchs[2:]]
                    else:
                        yield [[self._to_tensor(x) for x in batchs[0]]] + \
                                [self._to_tensor(x) for x in batchs[1:]]
            else:
                yield batch

        if eindex >= len(self.data):
            batch = self.data[sindex:]
            if self.to_tensor:
                if self.train:
                    batchs = self.dataset.pad_to_batch(batch, self.word2idx)
                    yield [self._to_tensor(x) for x in batchs]
                else:
                    batchs = self.dataset.pad_to_story(batch)
                    if self.return_masks:
                        yield [[self._to_tensor(y) for y in x] for x in batchs[:2]] + \
                                [self._to_tensor(x) for x in batchs[2:]]
                    else:
                        yield [[self._to_tensor(x) for x in batchs[0]]] + \
                                [self._to_tensor(x) for x in batchs[1:]]
            else:
                yield batch

    def __getitem__(self, idx):
        batchs = self.dataset.pad_to_story([self.data[idx]])
        if self.return_masks:
            return [[self._to_tensor(y) for y in x] for x in batchs[:2]] + \
                  [self._to_tensor(x) for x in batchs[2:]]
        else:
            return [[self._to_tensor(x) for x in batchs[0]]] + \
                  [self._to_tensor(x) for x in batchs[1:]]