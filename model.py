import torch
import pdb


class MultiTagger(torch.nn.Module):
    '''Tagger applies an RNN (LSTM or GRU) module and a Linear 
    module in sequence to an embedded sentence.'''


    def report_architecture(self):
        c = self.config
        print('-'*20+'\nMultiTagger Architecture:')
        print('\tEmbedding Size (n_embed,d_embed): %d,%d' % (c.n_embed,c.d_embed))
        print('\tEmbedding Dropout Ratio: %.2f' % c.emb_dp_ratio)
        rnntype = self.config.rnn
        if self.config.birnn:rnntype = 'Bidirectional '+rnntype
        rnndims = (self.rnn.input_size,self.rnn.hidden_size,self.rnn.num_layers)
        print('\tRNN Class: %s' % rnntype)
        print('\tRNN Dimensions (d_embed,d_hidden,n_layers): %d,%d,%d' % rnndims)
        print('\tRNN Dropout Ratio: %.2f' % self.config.rnn_dp_ratio)
        lineardims = (self.decoder.in_features,self.decoder.out_features)
        print('\tLinear Decoder (d_hidden,d_out): %d,%d' % lineardims)
        taskdout = ','.join(['d_out_'+str(x) for x in range(len(c.d_out))])
        taskdims = ','.join([str(x) for x in c.d_out])
        print(('\tTasks (%s): (%s) \n'+'-'*20) % (taskdout,taskdims))


    def init_hidden(self,bsize):
        n_l,d_h = self.config.n_layers,sum(self.config.d_hidden)
        if self.config.birnn:n_l *= 2      
        weight = next(self.parameters()).data
        if self.config.rnn == 'GRU':
            return torch.autograd.Variable(weight.new(n_l,bsize,d_h).zero_())
        else:
            return (torch.autograd.Variable(weight.new(n_l,bsize,d_h).zero_()),
                    torch.autograd.Variable(weight.new(n_l,bsize,d_h).zero_()))


    def init_weights(self,initrange = 0.1):
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)


    def __init__(self,c):
        super(MultiTagger,self).__init__()
        self.config = c
        self.dropout = torch.nn.Dropout(p = c.emb_dp_ratio)
        self.encoder = torch.nn.Embedding(c.n_embed,c.d_embed)
        if c.rnn == 'LSTM':rnnclass = torch.nn.LSTM
        elif c.rnn == 'GRU':rnnclass = torch.nn.GRU
        else:raise ValueError('... unknown RNN class: %s ...' % c.rnn)
        self.rnn = rnnclass(
            input_size = c.d_embed,
            hidden_size = sum(c.d_hidden),
            num_layers = c.n_layers,
            dropout = c.rnn_dp_ratio,
            bidirectional = c.birnn)
        decoder_isize = sum(c.d_hidden)*2 if c.birnn else sum(c.d_hidden)
        self.decoder = torch.nn.Linear(decoder_isize,sum(c.d_out))
        self.init_weights()
        self.report_architecture()


    def forward(self,batch):
        i = batch.__getattribute__(self.config.target_field)
        emb = self.dropout(self.encoder(i))
        o,h = self.rnn(emb,self.init_hidden(emb.size()[1]))
        decoded = self.decoder(o.view(o.size(0)*o.size(1),o.size(2)))
        scores = decoded.view(o.size(0),o.size(1),decoded.size(1))
        output = []
        u,w = 0,0
        for j in range(len(self.config.d_out)):
            v,z = self.config.d_out[j],self.config.d_hidden[j]
            if self.config.rnn == 'GRU':
                output.append((scores[:,:,u:u+v],h[:,:,w:w+z]))
            elif self.config.rnn == 'LSTM':
                output.append((scores[:,:,u:u+v],h[0][:,:,w:w+z],h[1][:,:,w:w+z]))
            else:raise ValueError
            u += v;w += z
        return output


