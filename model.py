import torch
from torch.autograd import Variable
from functools import reduce
import types,pdb


class Tagger(torch.nn.Module):
    '''Tagger applies an RNN (LSTM or GRU) module and a Linear 
    module in sequence to an embedded sentence.'''


    def report_architecture(self):
        rnntype = self.config.rnn
        if self.config.birnn:rnntype = 'Bidirectional '+rnntype
        rnndims = (self.rnn.input_size,self.rnn.hidden_size,self.rnn.num_layers)
        lineardims = (self.decoder.in_features,self.decoder.out_features)
        print('\tRNN Class: %s' % rnntype)
        print('\tRNN Dimensions (d_in,d_hidden,n_layers): %d,%d,%d' % rnndims)
        print('\tRNN Dropout Ratio: %.2f' % self.config.dp_ratio)
        print('\tLinear Decoder (d_in,d_out): %d,%d\n' % lineardims)


    def init_hidden(self,bsize):
        n_l,d_h = self.config.n_layers,self.config.d_hidden
        if self.config.birnn:n_l *= 2      
        weight = next(self.parameters()).data
        if self.config.rnn == 'GRU':
            return Variable(weight.new(n_l,bsize,d_h).zero_())
        else:
            return (Variable(weight.new(n_l,bsize,d_h).zero_()),
                    Variable(weight.new(n_l,bsize,d_h).zero_()))


    def __init__(self,c):
        super(Tagger,self).__init__()
        self.config = c
        if c.rnn == 'LSTM':rnnclass = torch.nn.LSTM
        elif c.rnn == 'GRU':rnnclass = torch.nn.GRU
        else:raise ValueError('... unknown RNN class: %s ...' % c.rnn)
        self.rnn = rnnclass(
            input_size = c.d_in,
            hidden_size = c.d_hidden,
            num_layers = c.n_layers,
            dropout = c.dp_ratio,
            bidirectional = c.birnn)
        decoder_isize = c.d_hidden*2 if c.birnn else c.d_hidden
        self.decoder = torch.nn.Linear(decoder_isize,c.d_out)


    def forward(self,emb):
        h = self.init_hidden(emb.size()[1])
        o,h = self.rnn(emb,h)
        decoded = self.decoder(o.view(o.size(0)*o.size(1),o.size(2)))
        scores = decoded.view(o.size(0),o.size(1),decoded.size(1))
        return scores,h


class MultiTagger(torch.nn.Module):
    '''MultiTagger applies a set of independent Tagger modules in parallel.
    Each Tagger is given the same embedded sentence.'''


    def report_architecture(self):
        c = self.config
        print('-'*20+'\nMultiTagger architecture:')
        print('\tEmbedding Size (n_embed,d_embed): %d,%d' % (c.n_embed,c.d_embed))
        print('\tEmbedding Dropout Ratio: %.2f' % c.emb_dp_ratio)
        print(('\tNumber of Taggers: %d \n'+'-'*20+'\n') % c.n_taggers)
        for j,t in enumerate(self.taggers):
            print('-- Tagger %d --' % (j+1))
            t.report_architecture()
        print('-'*20)


    def init_weights(self,initrange = 0.1):
        self.encoder.weight.data.uniform_(-initrange,initrange)
        for t in self.taggers:
            t.decoder.bias.data.fill_(0)
            t.decoder.weight.data.uniform_(-initrange,initrange)


    def __init__(self,c):
        super(MultiTagger,self).__init__()
        self.config = c
        self.dropout = torch.nn.Dropout(p = c.emb_dp_ratio)
        self.encoder = torch.nn.Embedding(c.n_embed,c.d_embed)
        self.taggers = []
        c.d_in = [c.d_embed]*c.n_taggers
        if not c.notnested:
            d_out_mods = reduce(lambda x,y : x+[x[-1]+y],[[0]]+c.d_out)
            c.d_in = [x+y for x,y in zip(c.d_in,d_out_mods)]
        tagattrs = ('rnn','d_in','d_hidden','n_layers','dp_ratio','birnn','d_out')
        for j in range(c.n_taggers):
            tc = types.SimpleNamespace()
            for k in tagattrs:
                tc.__setattr__(k,c.__getattribute__(k)[j])
            self.taggers.append(Tagger(tc))
        self.init_weights()
        self.report_architecture()


    def forward(self,batch):
        i = batch.__getattribute__(self.config.target_field)
        emb = self.dropout(self.encoder(i))
        if self.config.notnested:
            return tuple(t(emb) for t in self.taggers)
        else:
            i = emb
            outputs = []
            for t in self.taggers:
                o,h = t(i)
                outputs.append((o,h))
                i = torch.cat([i,o],2)
            return outputs


