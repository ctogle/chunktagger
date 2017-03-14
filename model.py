import torch
from torch.autograd import Variable
from functools import reduce
import pdb


class Tagger(torch.nn.Module):
    '''Tagger applies an LSTM module and a Linear 
    module in sequence to an embedded sentence.'''


    def init_hidden(self,bsize):
        n_l,d_h = self.config.n_layers,self.config.d_hidden
        if self.config.birnn:n_l *= 2      
        weight = next(self.parameters()).data
        return (Variable(weight.new(n_l,bsize,d_h).zero_()),
                Variable(weight.new(n_l,bsize,d_h).zero_()))


    def __init__(self,c,j):
        super(Tagger,self).__init__()
        self.config = c
        isize = c.d_embed if c.notnested else c.d_embed+c.d_out_mods[j]
        self.rnn = torch.nn.LSTM(
            input_size = isize,
            hidden_size = c.d_hidden,
            num_layers = c.n_layers,
            dropout = c.dp_ratio,
            bidirectional = c.birnn)
        decoder_isize = c.d_hidden*2 if c.birnn else c.d_hidden
        self.decoder = torch.nn.Linear(decoder_isize,c.d_outs[j])


    def forward(self,emb):
        h = self.init_hidden(emb.size()[1])
        o,h = self.rnn(emb,h)
        decoded = self.decoder(o.view(o.size(0)*o.size(1),o.size(2)))
        scores = decoded.view(o.size(0),o.size(1),decoded.size(1))
        return scores,h


class MultiTagger(torch.nn.Module):
    '''MultiTagger applies a set of independent Tagger modules in parallel.
    Each Tagger is given the same embedded sentence.'''


    def init_weights(self,initrange = 0.1):
        self.encoder.weight.data.uniform_(-initrange,initrange)
        for t in self.taggers:
            t.decoder.bias.data.fill_(0)
            t.decoder.weight.data.uniform_(-initrange,initrange)


    def __init__(self,c):
        super(MultiTagger,self).__init__()
        self.config = c
        self.dropout = torch.nn.Dropout(p = c.dp_ratio)
        self.encoder = torch.nn.Embedding(c.n_embed,c.d_embed)
        c.d_out_mods = reduce(lambda x,y : x+[x[-1]+y],[[0]]+c.d_outs)
        self.taggers = tuple(Tagger(c,x) for x in range(c.n_taggers))
        self.init_weights()


    def forward(self,batch):
        emb = self.encoder(batch.sentence)
        #emb = self.encoder(batch.reversal)
        emb = self.dropout(emb)
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


