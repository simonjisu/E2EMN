import torch
import torch.nn as nn



class MEMN2N(nn.Module):
    def __init__(self, vocab_size, embed_size, n_hops=3, weight_style='adjacent',
                 encoding_method='bow', temporal=True, maxlen_story=None, pad_idx=0):
        """
        https://arxiv.org/pdf/1503.08895.pdf
        Args:
        - vocab_size: length of vocabulary
        - embed_size: size of embedding dimension  
        - n_hops: multiple computational steps for memories
        - weight_style: 
            * 'adjacent': share all weights B = A(1) = C(1) = ... = C(K) = W^T
            * 'rnnlike': share weights 
        - encoding_method:
        - temporal:
        - maxlen_story:
        - pad_idx:
        """
        super(MEMN2N, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.n_hops = n_hops
        self.maxlen_story = maxlen_story
        self.weight_style = weight_style.lower()
        self.encoding_method = encoding_method.lower()
        self.te = temporal
        self.pad_idx = pad_idx
        
        self.layers_init()
        self.apply(self.weight_init)
        
    def layers_init(self):
        """
        two types: adjacent, rnnlike
        """
        self.context_modules = nn.ModuleDict([('embedding_{}_{}'.format(n, k), 
                                               nn.Embedding(self.vocab_size, 
                                                            self.embed_size,
                                                            padding_idx=self.pad_idx)) \
                                              for k, n in enumerate(['A', 'C']*self.n_hops)])
        # adjacent weight sharing style
        if self.weight_style == 'adjacent':
            for name, mod in self.context_modules.items():
                idx = int(name.split('_')[-1])
                if idx == 0:
                    self.embedding_B = nn.Embedding(self.vocab_size, 
                                                    self.embed_size, 
                                                    padding_idx=self.pad_idx)
                    self.embedding_B.weight.data = mod.weight.data 
                elif idx % 2 == 0:
                    mod.weight.data = self.context_modules['embedding_C_{}'.format(idx-1)].weight.data
                elif idx == (self.n_hops*2-1):
                    self.linear_final = nn.Linear(self.embed_size, self.vocab_size, bias=False)
                    self.linear_final.weight.data = mod.weight.data
            
        # rnn-like weight sharing style        
        elif self.weight_style == 'rnnlike':
            for name, mod in self.context_modules.items():
                idx = int(name.split('_')[-1])
                if idx <= 1 :
                    continue
                elif idx % 2 == 0:
                    mod.weight.data = self.context_modules['embedding_A_0'].weight.data
                else:
                    mod.weight.data = self.context_modules['embedding_C_1'].weight.data

            # others layers
            self.embedding_B = nn.Embedding(self.vocab_size, 
                                            self.embed_size, 
                                            padding_idx=self.pad_idx)
            self.linear_mapping = nn.Linear(self.embed_size, self.embed_size)  # inear mapping for each hops
            self.linear_final = nn.Linear(self.embed_size, self.vocab_size, bias=False)
        else:
            assert True, 'Insert "adjacent" or "rnnlike" in weight_style'
        # common
        self.softmax_layer = nn.Softmax(dim=1)
        if self.te:
            self.temporal_modules = nn.ModuleDict([('temporal_{}'.format(n), 
                                                    nn.Embedding(self.maxlen_story+1, 
                                                                 self.embed_size, 
                                                                 padding_idx=self.pad_idx)) \
                                                   for n in ['A', 'C']])
    def weight_init(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight.data, mean=0, std=0.1)
            m.weight.data[0] = 0
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, mean=0, std=0.1)
            if self.weight_style == 'adjacent':
                m.weight.data[0] = 0
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)
                
    def encoding2memory(self, embeded_x):
        sum_dim = 2 if embeded_x.dim() == 4 else 1  # stories sum_dim=2, queries sum_dim=1
        *_, len_words, embed_size = embeded_x.size()  # len_words=J, embed_size=d
        
        if self.encoding_method == 'bow':
            return embeded_x.sum(sum_dim)
        
        if self.encoding_method == 'pe':
            temp = torch.ones_like(embeded_x, device='cpu')
            k = temp * torch.arange(1, embed_size+1, dtype=torch.float) / embed_size
            l = temp * torch.arange(1, len_words+1, dtype=torch.float).unsqueeze(1) / len_words
            position = (1- l) - k * (1 - 2*l)
            position_encoded = (embeded_x * position.to(embeded_x.device)).sum(sum_dim)
            return position_encoded  # B, (len_story), embed_size
        
        else:
            assert True, 'Insert "bow" or "pe" in encoding_method'
            
    def forward(self, stories, queries, stories_idx, ls=False, return_p=False):
        """
        Inputs:
        - stories: B, maxlen_story(T), maxlen_words(n)
        - queries: B, maxlen_query(T_q)
        - stories_idx: B, maxlen_story(T)
        - ls: linear start
        Outputs:
        - log softmaxed score(nll loss), if return_p=False else return answer score and p
        """
        ps = []
        # Start Learning
        embeded_b = self.embedding_B(queries) 
        u_next = self.encoding2memory(embeded_b)  # (B, d)
        
        for k in range(self.n_hops):
            embeded_a = self.context_modules['embedding_A_{}'.format(2*k)](stories)  # (B, T, n, d)
            embeded_c = self.context_modules['embedding_C_{}'.format(2*k+1)](stories)  # (B, T, n, d)
            m = self.encoding2memory(embeded_a)  # (B, T, d)
            c = self.encoding2memory(embeded_c)  # (B, T, d)
            if self.te:
                m += self.temporal_modules['temporal_A'](stories_idx)  # (B, T, d)
                c += self.temporal_modules['temporal_C'](stories_idx)  # (B, T, d)
            if ls:
                # (B, T, d) x (B, d, 1) = (B, T, 1)
                p = torch.bmm(m, u_next.unsqueeze(2))
            else:
                p = self.softmax_layer(torch.bmm(m, u_next.unsqueeze(2)))
                ps.append(p.squeeze(2))  # [(B, T) for all hops]
                
            o = (c * p).sum(1)  # (B, T, d) * (B, T, 1) = (B, T, d) > (B, d)
            
            if self.weight_style == 'rnnlike':
                # (B, d) + (B, d) = (B, d)
                u_next = self.linear_mapping(u_next) + o
            else:
                u_next = u_next + o
                
        a = self.linear_final(u_next)  # (B, d) > (B, V)
        if return_p:
            return a, ps
        return torch.log_softmax(a, dim=1)  # use nll loss