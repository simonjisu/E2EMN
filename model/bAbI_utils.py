# -*- coding utf-8 -*-
import torch
from torch.autograd import Variable
from copy import deepcopy
import numpy as np
import random
flatten = lambda l: [item for sublist in l for item in sublist]


def bAbI_data_loader(path, vocab=None):
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
                a = a.lower().strip().split() + ['</s>']
                support = int(support)
                story_temp = deepcopy(story)
                data_temp.append([story_temp, q, a, support])
            else:
                sentence = line.lower().replace('.', '').split() + ['</s>']
                story.append(sentence)

    except:
        print('check data')
        return None

    if vocab:
        data, vocab = bAbI_build_vocab(data_temp, vocab)
    else:
        data, vocab = bAbI_build_vocab(data_temp)

    return data, vocab


def bAbI_build_vocab(data, vocab=None):
    if vocab is None:
        story, q, a, s = list(zip(*data))
        vocab = list(set(flatten(flatten(story)) + flatten(q) + flatten(a)))
        word2idx = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        for word in vocab:
            if word2idx.get(word) is None:
                word2idx[word] = len(word2idx)
        idx2word = {v: k for k, v in word2idx.items()}
    else:
        word2idx = vocab

    for d in data:
        # d[0]: stories
        # d[1]: questions
        # d[2]: answer
        # d[3]: support
        for i, story in enumerate(d[0]):
            d[0][i] = transfer2idx(story, word2idx)

        d[1] = transfer2idx(d[1], word2idx)
        d[2] = transfer2idx(d[2], word2idx)

    return data, word2idx


def transfer2idx(seq, dictionary):
    idxs = list(map(lambda w: dictionary[w] if dictionary.get(w) is not None else \
                    dictionary["<unk>"], seq))
    return idxs


def data_loader(data, batch_size, shuffle=False):
    if shuffle: random.shuffle(data)
    sindex = 0
    eindex = batch_size
    while eindex < len(data):
        batch = data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(data):
        batch = data[sindex:]
        yield batch


def pad_to_batch(batch, w2idx, no_batch=False):
    """
    stories, stories_masks: B, n, T_c
    questions, questions_masks: B, T_q
    answers: B, T_a
    supports: B
    -------------------------------------
    output: stories, stories_masks, questions, questions_masks, answers, supports
    """
    story, q, a, s = list(zip(*batch))
    max_story = max([len(s) for s in story])  # max_stories
    max_len = max([len(s) for s in flatten(story)])  # max_sentence_len
    max_q = max([len(q_) for q_ in q])
    max_a = max([len(a_) for a_ in a])

    stories, stories_masks = [], []
    for i in range(len(batch)):
        story_array, story_mask = get_batch_array(get_fixed_array(story[i], w2idx), no_batch, max_story, max_len)
        stories.append(story_array)
        stories_masks.append(story_mask)

    questions, questions_masks = get_batch_array(get_fixed_array(q, w2idx), no_batch, len(batch), max_q)
    answers, _ = get_batch_array(get_fixed_array(a, w2idx), no_batch, len(batch), max_a)
    return [trans2tensor(x) for x in [stories, stories_masks, questions, questions_masks, answers]] + [s]


def get_fixed_array(data, w2idx):
    max_col = max([len(d) for d in data])
    for j in range(len(data)):
        if len(data[j]) < max_col:
            data[j].append(w2idx.get('<pad>'))
    return data


def get_batch_array(data, no_batch, *shape):
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


def trans2tensor(x):
    return Variable(torch.LongTensor(x))


def pad_to_story(test_data, w2idx, no_batch=True):  # this is for inference
    """
    output: stories, stories_masks, questions, questions_masks, answers, supports
    """
    story, q, a, s = list(zip(*test_data))
    max_story = max([len(s) for s in story])  # max_stories
    max_len = max([len(s) for s in flatten(story)])  # max_sentence_len
    max_q = max([len(q_) for q_ in q])
    max_a = max([len(a_) for a_ in a])

    stories, stories_masks = [], []
    for i in range(len(test_data)):
        story_array, story_mask = get_batch_array(get_fixed_array(story[i], w2idx), no_batch, max_story, max_len)
        stories.append(story_array)
        stories_masks.append(story_mask)

    questions, questions_masks = get_batch_array(get_fixed_array(q, w2idx), no_batch, len(test_data), max_q)
    answers, _ = get_batch_array(get_fixed_array(a, w2idx), no_batch, len(test_data), max_a)

    stories = [trans2tensor(x) for x in stories]
    stories_masks = [trans2tensor(x) for x in stories_masks]

    return [stories, stories_masks] + [trans2tensor(x) for x in [questions, questions_masks, answers]] + [s]