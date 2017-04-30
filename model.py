import torch
import numpy,functools,os,time,pdb
import util


class MultiTagger(torch.nn.Module):
    '''Tagger applies an RNN (LSTM or GRU) module and a Linear 
    module in sequence to an embedded sentence.
    
    It uses a concatenated hidden layer to perform multiple tagging operations 
    that share some dependence (e.g. part of speech and chunking).
    '''


    def report_architecture(self):
        c = self.config
        print('-'*20+'\nMultiTagger Architecture:')
        print('\tSaved At: %s' % c.modelcache)
        print('\tEmbedding Size (n_embed,d_embed): %d,%d' % (c.n_embed,c.d_embed))
        print('\tEmbedding Dropout Ratio: %.2f' % c.emb_dp_ratio)
        rnntype = self.config.rnn
        if self.config.birnn:rnntype = 'Bidirectional '+rnntype
        rnndims = (self.rnn.input_size,self.rnn.hidden_size,self.rnn.num_layers)
        print('\tRNN Class: %s' % rnntype)
        print('\tRNN Dimensions (d_embed,d_hidden,n_layers): %d,%d,%d' % rnndims)
        hidstr = ','.join([str(d) for d in c.d_hidden])
        print(('\tRNN d_hidden breakdown: (%s)') % hidstr)
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
        if type(c.d_hidden) == type(''):
            c.d_hidden = tuple(int(v) for v in c.d_hidden.split(','))
        if not len(c.d_hidden) == c.n_taggers:
            amsg = '... received %d d_hidden entries; require %d ...'
            raise ValueError(amsg % (len(c.d_hidden),c.n_taggers))
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
        c.training_accuracy = -1.0


    def forward(self,i):
        '''Input i is a padded sentence LongTensor (time,batchsize)'''
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


    def work(self,s):
        '''Perform tagging of a raw sentence.'''
        inputs,answers = self.config.fields
        t = (s,'.',',','?','\'')
        t = functools.reduce(lambda s,c : s.replace(c,' '+c),t).split(' ')
        n = numpy.array([inputs.vocab.stoi[w] for w in t])
        i = torch.from_numpy(n).view(n.shape[0],1)
        o = self.forward(torch.autograd.Variable(i))
        o = [torch.max(x[0],2)[1].view(n.shape[0]).data for x in o]
        o = [[a.vocab.itos[y] for y in x] for a,x in zip(answers,o)]
        o = dict([(k,x) for k,x in zip(self.config.output_fields,o)])
        o[self.config.target_field] = t
        return o





'''For summing losses across sentences'''
sloss = lambda c,a,b : sum([c(a[:,i],b[:,i]) for i in range(a.size()[1])])
def train_batch(tagger,crit,opt,batch,bdataf):
    '''Perform training on a single batch of examples, 
    returning the number of correct answers'''
    opt.zero_grad()
    i = batch.__getattribute__(tagger.config.target_field)
    tdata,bdata = tagger(i),bdataf(batch)
    loss = sum([sloss(crit,o[0],c) for o,c in zip(tdata,bdata)])
    loss.backward();opt.step()
    return [util.ncorrect(o[0],c) for o,c in zip(tdata,bdata)]


def train_epoch(tagger,crit,opt,batcher,epoch,stime,bdataf,prog_f):
    '''Perform a training epoch given an iterator of training batches,
    returning the accuracy of the model on the data set.'''
    if prog_f:prog_f(epoch,-1,len(batcher),[-1]*tagger.config.n_taggers,0,stime)
    batcher.init_epoch()
    corrects,total = [0]*tagger.config.n_taggers,0
    for j,batch in enumerate(batcher):
        newcorrects = train_batch(tagger,crit,opt,batch,bdataf)
        corrects = [x+y for x,y in zip(corrects,newcorrects)]
        total += batch.batch_size*bdataf(batch)[0].size()[0]
        if prog_f:prog_f(epoch,j,len(batcher),corrects,total,stime)
        if time.time()-stime > tagger.config.timeout:raise KeyboardInterrupt
    return 100.0*sum(corrects)/(total*tagger.config.n_taggers)


def train(tagger,train_i,test_i,bdataf,prog_h,prog_f):
    '''Perform training of the model given an iterator of training batches.
    Exit the training process early on KeyboardInterrupt, or if accuarcy 
    improvement is sufficiently slow.
    Save the model between training epochs or upon early exit.
    Test the accuracy of the model on an iterator of test batches when 
    training is complete.'''
    config = tagger.config
    lossf = torch.nn.CrossEntropyLoss()
    if hasattr(torch.optim,config.optimizer):
        optclass = torch.optim.__getattribute__(config.optimizer)
        opt = optclass(tagger.parameters(),lr = config.lr)
    else:raise ValueError('... unavailable optimizer: %s ...' % config.optimizer)
    test_required = False
    improvement_threshold = 0.0
    if not config.training_accuracy < config.targetaccuracy and prog_f:
        print(prog_h)
    stime = time.time()
    for j in range(config.epochs):
        try:
            if not config.training_accuracy < config.targetaccuracy:
                print('... target accuracy has been met ... ending training ...')
                break
            accuracy = train_epoch(tagger,lossf,opt,train_i,j,stime,bdataf,prog_f)
            improvement = accuracy-config.training_accuracy
            config.training_accuracy = accuracy
            test_required = True
            torch.save(tagger,config.modelcache)
            if improvement < improvement_threshold:
                print('... improvement is quite low ... ending training ...')
                break
        except KeyboardInterrupt:
            if prog_f:prog_f(j,0,1,[-1]*config.n_taggers,0,stime)
            print('... training forcefully exited ...')
            torch.save(tagger,config.modelcache)
            break
    return test_required


def test(tagger,batcher,bdataf,prog_f):
    '''Perform testing given an iterator of testing batches,
    returning the accuracy of the model on the data set.'''
    config = tagger.config
    stime = time.time()
    prog_f(0,-1,len(batcher),[-1]*config.n_taggers,0,stime)
    tagger.eval();batcher.init_epoch()
    corrects,total = [0]*config.n_taggers,0
    for j,batch in enumerate(batcher):
        i = batch.__getattribute__(tagger.config.target_field)
        tdata,bdata = tagger(i),bdataf(batch)
        newcorrects = [util.ncorrect(o[0],c) for o,c in zip(tdata,bdata)]
        corrects = [x+y for x,y in zip(corrects,newcorrects)]
        total += batch.batch_size*bdataf(batch)[0].size()[0]
        prog_f(0,j,len(batcher),corrects,total,stime)
    return 100.0*sum(corrects)/(total*config.n_taggers)


def newmodel(config,data):
    '''Create or load, train, and test an instance of a MultiTagger.'''
    train_i,test_i = data['dataset_iters']
    inputs,answers = data['fields']
    fkeys = train_i.dataset.fields.keys()
    bkeys = [k for k in fkeys if train_i.dataset.fields[k] in answers]
    bdataf = lambda b : tuple(b.__getattribute__(s) for s in bkeys)
    config.output_fields = tuple(bkeys)
    config.n_taggers = len(bkeys)
    if not config.fresh and os.path.exists(config.modelcache):
        if config.gpu >= 0:
            map_location = lambda storage,locatoin : storage.cuda(config.gpu)
        else:map_location = lambda storage,location : None
        tagger = torch.load(config.modelcache,map_location = map_location)
        tagger.config.targetaccuracy = config.targetaccuracy
        print('... loaded cached model: %s ...' % config.modelcache)
    else:
        tagger = MultiTagger(config)
        if config.word_vectors:
            tagger.encoder.weight.data = inputs.vocab.vectors
            if config.gpu >= 0 and torch.cuda.is_available():tagger.cuda()
        print('... created new model (%s) ...' % config.modelcache)
        if config.epochs == 0:
            print('... new model requires training ...')
            print('... resetting epochs from 0 to 100 ...')
            config.epochs = 100
    tagger.report_architecture()

    prog_h,prog_f = util.get_progress_function(bkeys)

    if train_i and config.epochs:
        print('... training model ...')
        tagger.train()
        test_required = train(tagger,train_i,test_i,bdataf,prog_h,prog_f)

    tagger.eval()
    if test_i and test_required:
        print('... testing model ...')
        if prog_f:print(prog_h)
        accuracy = test(tagger,test_i,bdataf,prog_f)
        tagger.config.testing_accuracy = accuracy
        torch.save(tagger,config.modelcache)
        print('... final model task-averaged accuracy: %.2f ...' % accuracy)

    tagger.config.fields = (inputs,answers)
    return tagger


