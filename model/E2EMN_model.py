# -*- coding utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class E2EMN(nn.Module):
    def __init__(self, vocab_size, embed_size, n_hops=3, encoding_method='basic', temporal=True, \
                 use_cuda=False, max_story_len=None):
        """
        https://arxiv.org/pdf/1503.08895.pdf

        ------------------------------------
        vocab_size: [int], vocaburary size
        embed_size: [int], embedding size
        n_hops: [int], multiple computational steps
        encoding_method: [string], "basic" or "pe", "pe" means position encoding
        temporal: [boolean], Use temporal encoding method
        use_cuda: [boolean], GPU option
        max_story_len: [int], max story length in total data set, it is used for determining embedding size
                       of temporal encoding. Find it at "bAbIDataset" class, self.max_story_len
        """
        super(E2EMN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_hops = n_hops
        self.encoding_method = encoding_method.lower()
        self.te = temporal
        self.use_cuda = use_cuda

        # sharing matrix for k hops & and init to normal dist.
        self.embed_A = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_B = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_C = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        # TE: temporal encoding
        if self.te:
            assert max_story_len is not None, \
                'must have a fixed story_len, insert "max_story_len" as a number, find it at "bAbIDataset" class'
            assert isinstance(max_story_len, int), '"max_story_len" must be a integer'

            self.embed_A_T = nn.Embedding(max_story_len + 1, self.embed_size, padding_idx=0)
            self.embed_C_T = nn.Embedding(max_story_len + 1, self.embed_size, padding_idx=0)
            if self.use_cuda:
                self.embed_A_T = self.embed_A_T.cuda()
                self.embed_C_T = self.embed_C_T.cuda()

        self.linear = nn.Linear(embed_size, vocab_size)
        self._weight_init()

    def _weight_init(self):
        for x in [self.embed_A, self.embed_B, self.embed_C]:
            nn.init.normal(x.weight, mean=0, std=0.1)
        if self.te:
            for x in [self.embed_A_T, self.embed_C_T]:
                nn.init.normal(x.weight, mean=0, std=0.1)

    def _temporal_encoding_requirements(self, stories_masks):
        # temporal encoding
        if self.te:
            story_len = stories_masks.size(1)
            temp = stories_masks.eq(0).sum(2)  # B, n : byte tensor
            te_idx_matrix = Variable(torch.arange(1, story_len + 1).repeat(temp.size(0)).view(temp.size()), \
                                     requires_grad=False).long()
            if self.use_cuda:
                te_idx_matrix = te_idx_matrix.cuda()
            te_idx_matrix = te_idx_matrix * temp.ge(1).long()  # B, n
        else:
            te_idx_matrix = None

        return te_idx_matrix

    def _pe_requirements(self, stories_masks):
        # position encoding
        if stories_masks is not None:
            pe_word_lengths = stories_masks.eq(0).sum(2)  # B, n : byte tensor
        else:
            pe_word_lengths = None
        return pe_word_lengths

    def encoding2memory(self, embeded_x, word_length=None):
        """
        embed_x: n, T_c, d
        word_length: n
        """
        if self.encoding_method == 'basic':
            return embeded_x.sum(1)  # n, d

        elif self.encoding_method == 'pe':
            assert word_length is not None, 'insert stories_masks when forward'

            T_c, d = embeded_x.size()[1:]
            j = Variable(torch.arange(1, T_c + 1).unsqueeze(1).repeat(1, d), requires_grad=False)
            k = Variable(torch.arange(1, d + 1).unsqueeze(1).repeat(1, T_c).t(), requires_grad=False)
            if self.use_cuda:
                j, k = j.cuda(), k.cuda()

            embeded_x_pe = []
            for embed, J in zip(embeded_x, word_length.float()):  # iteration of n size
                # embed: T_c d
                # J: scalar
                if J.eq(0).data[0]:  # all words are pad data, which means word_length = 0
                    embeded_x_pe.append(embed)
                else:
                    l = (torch.ones_like(embed).float() - j / J) - (k / d) * (torch.ones_like(embed) - (2 * j) / J)
                    embed = embed * l
                    embeded_x_pe.append(embed)  # T_c, d
            embeded_x_pe = torch.stack(embeded_x_pe)  # n, T_c, d
            return embeded_x_pe.sum(1)  # n, d

        else:
            assert True, 'insert encoding_method key value in the model, default is "basic".'

    def forward(self, stories, questions, stories_masks=None, questions_masks=None):
        """
        stories, stories_masks: B, n, T_c
        questions, questions_masks: B, T_q
        """
        # init some requirements
        te_idx_matrix = self._temporal_encoding_requirements(stories_masks)  # B, n
        pe_word_lengths = self._pe_requirements(stories_masks)  # B, n

        # Start Learning
        o_list = []
        # questions: B, T_q
        embeded_B = self.embed_B(questions)  # B, T_q, d
        u = embeded_B.sum(1)  # u: B, d
        o_list.append(u)  # [(B, d)]

        for k in range(self.n_hops):
            # encoding part: PE, TE
            batch_memories = []  # B, n, d
            batch_contexts = []  # B, n, d
            for i, inputs in enumerate(stories):  # iteration of batch
                # inputs: n, T_c
                embeded_A = self.embed_A(inputs)  # n, T_c, d
                embeded_C = self.embed_C(inputs)  # n, T_c, d
                # basic or PE
                m = self.encoding2memory(embeded_A, pe_word_lengths[i])  # n, d
                c = self.encoding2memory(embeded_C, pe_word_lengths[i])  # n, d
                # TE
                if self.te:
                    A_T = self.embed_A_T(te_idx_matrix[i])  # n, d
                    C_T = self.embed_C_T(te_idx_matrix[i])  # n, d
                    m = m + A_T
                    c = c + C_T
                batch_memories.append(m)
                batch_contexts.append(c)

            batch_memories = torch.stack(batch_memories)  # B, n, d
            batch_contexts = torch.stack(batch_contexts)  # B, n, d

            # attention part: select which sentence to attent
            # score = m * u[-1] : (B, n, d) * (B, d, 1) = B, n, 1
            score = torch.bmm(batch_memories, o_list[-1].unsqueeze(2))
            probs = F.softmax(score, dim=1)  # p: B, n, 1

            # output: element-wies mul & sum (B, n, d) x (B, n, 1) = B, n, d --> B, d
            o = torch.sum(batch_contexts * probs, 1)

            o_next = o_list[-1] + o
            o_list.append(o_next)  # B, d

        # guessing part:
        outputs = self.linear(o_list[-1])  # B, d > B, V
        return outputs