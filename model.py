import torch
import pdb


class POSTagger(torch.nn.Module):


    def init_weights(self,initrange = 0.1):
        self.encoder.weight.data.uniform_(-initrange,initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange,initrange)


    def init_hidden(self,bsz):
        n_l,d_h = self.config.n_layers,self.config.d_hidden
        # n_l*=2 if self.birnn?
        weight = next(self.parameters()).data
        return (torch.autograd.Variable(weight.new(n_l,bsz,d_h).zero_()),
                torch.autograd.Variable(weight.new(n_l,bsz,d_h).zero_()))


    def __init__(self,c):
        super(POSTagger,self).__init__()
        self.config = c
        dp_ratio,n_embed,d_embed,d_hidden,n_layers,birnn,d_out =\
            c.dp_ratio,c.n_embed,c.d_embed,c.d_hidden,c.n_layers,c.birnn,c.d_out

        self.dropout = torch.nn.Dropout(p = dp_ratio)
        self.encoder = torch.nn.Embedding(n_embed,d_embed)
        self.rnn = torch.nn.LSTM(
            input_size = d_embed,hidden_size = d_hidden,num_layers = n_layers,
            dropout = dp_ratio,bidirectional = birnn)
        decoder_isize = d_hidden*2 if birnn else d_hidden
        self.decoder = torch.nn.Linear(decoder_isize,d_out)
        self.init_weights()


    def forward(self,batch):
        # apply dropout to emb?
        emb = self.encoder(batch.sentence)
        hidden = self.init_hidden(emb.size()[1])
        output,hidden = self.rnn(emb,hidden)
        # apply dropout to output?
        #output = self.dropout(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1),output.size(2)))
        scores = decoded.view(output.size(0),output.size(1),decoded.size(1))
        return scores,hidden


