#!/home/cogle/anaconda3/bin/python3.6
import torch,torchtext
import os,time,pdb
import util,dataset,model


def fields():
    '''Create field objects and train and test data sets.
    Use the datasets to initialize the vocab objects of the fields.'''
    inputs = torchtext.data.Field(lower = config.lower)
    answers = [torchtext.data.Field() for x in range(config.n_taggers)]
    dsets = dataset.POSTags.splits(inputs,answers)
    inputs.build_vocab(*dsets)
    for a in answers:a.build_vocab(dsets[0])
    if config.word_vectors:
        if os.path.isfile(config.vectorcache):
            inputs.vocab.vectors = torch.load(config.vectorcache)
        else:
            inputs.vocab.load_vectors(
                wv_dir = config.cachedir,
                wv_type = config.word_vectors,
                wv_dim = config.d_embed)
            os.makedirs(os.path.dirname(config.vectorcache),exist_ok = True)
            torch.save(inputs.vocab.vectors,config.vectorcache)
    kws = {
        'batch_size' : config.batch_size,
        'device' : config.gpu,
        'sort' : False,'repeat' : False,
            }
    train_iter,test_iter = torchtext.data.BucketIterator.splits(dsets,**kws)
    return inputs,answers,train_iter,test_iter


def newmodel(word_vectors):
    '''Create or load an instance of the model.'''
    if not config.fresh and os.path.exists(config.modelcache):
        if config.gpu >= 0:
            map_location = lambda storage,locatoin : storage.cuda(config.gpu)
        else:map_location = lambda storage,location : None
        tagger = torch.load(config.modelcache,map_location = map_location)
        print('... loaded cached model: %s ...' % config.modelcache)
    else:
        tagger = model.MultiTagger(config)
        if config.word_vectors:
            tagger.encoder.weight.data = word_vectors
            if config.gpu >= 0 and torch.cuda.is_available():tagger.cuda()
        print('... created new model ...')
    return tagger


def train_batch(tagger,crit,opt,batch,v = False):
    '''Perform training on a single batch of examples, 
    returning the number of correct answers'''
    tagger.train();opt.zero_grad()
    bdata = batch.postags,batch.chunks
    tdata = tagger(batch)
    loss = sum([sloss(crit,o[0],c) for o,c in zip(tdata,bdata)])
    loss.backward();opt.step()
    if v:print(sprint(flatten([spair(o,c) for o,c in zip(tdata,bdata)])))
    return sum([ncorrect(o[0],c) for o,c in zip(tdata,bdata)])


def train_epoch(tagger,crit,opt,batcher):
    '''Perform a training epoch given an iterator of training batches,
    returning the accuracy of the model on the data set.'''
    batcher.init_epoch()
    correct,total = 0,0
    for j,batch in enumerate(batcher):
        bdata = batch.postags,batch.chunks
        correct += train_batch(tagger,crit,opt,batch,j == 0)
        total += sum([batch.batch_size*b.size()[0] for b in bdata])
        progress(j,len(batcher),100.0*correct/total)
    return 100.0*correct/total


def train(tagger,train_batcher,test_batcher):
    '''Perform training of the model given an iterator of training batches.
    Exit the training process early on KeyboardInterrupt, or if accuarcy 
    improvement is sufficiently slow.
    Save the model between training epochs or upon early exit.
    Test the accuracy of the model on an iterator of test batches when 
    training is complete.'''
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(tagger.parameters(),lr = config.learningrate)
    lastaccuracy = 0.0
    improvement_threshold = 0.01
    for j in range(config.epochs):
        try:
            stime = time.time()
            print('... begin epoch %i ...' % (j+1))
            accuracy = train_epoch(tagger,criterion,opt,train_batcher)
            print('... epoch %i took %.2f seconds ...' % (j+1,time.time()-stime))
            print('... model accuracy: %.2f ...' % accuracy)
            improvement = accuracy-lastaccuracy
            lastaccuracy = accuracy
            print('... accuracy / improvement: %.2f / %.2f ...' % (accuracy,improvement))
            if improvement < improvement_threshold:
                print('... improvement is quite low ... ending training ...')
                break
            elif improvement > 0.0:torch.save(tagger,config.modelcache)
        except KeyboardInterrupt:
            print('... training forcefully exited ...')
            torch.save(tagger,config.modelcache)
            break
    print('... testing accuracy ...')
    accuracy = test(tagger,test_batcher)
    print('... test accuracy: %.2f ...' % accuracy)


def test_batch(tagger,batch,v = False):
    '''Perform testing on a single batch of test examples,
    returning the number of correct answers.'''
    tagger.eval()
    bdata = batch.postags,batch.chunks
    tdata = tagger(batch)
    if v:print(sprint(flatten([spair(o,c) for o,c in zip(tdata,bdata)])))
    return sum([ncorrect(o[0],c) for o,c in zip(tdata,bdata)])


def test(tagger,batcher):
    '''Perform testing given an iterator of testing batches,
    returning the accuracy of the model on the data set.'''
    tagger.eval();batcher.init_epoch()
    correct,total = 0,0
    for j,batch in enumerate(batcher):
        bdata = batch.postags,batch.chunks
        correct += test_batch(tagger,batch,j == 0)
        total += sum([batch.batch_size*b.size()[0] for b in bdata])
        progress(j,len(batcher),100.0*correct/total)
    return 100.0*correct/total


def work(tagger,inputs,answers):
    '''As an example of totally distinct data usage, create a dataset of
    Wikipedia page sentences, a single batch for all of the sentences, 
    and run the model on them.'''
    dset = dataset.WikiData.splits(inputs,answers)[0]
    batch = torchtext.data.Batch(dset.examples,dset,config.gpu,False)
    tagger.eval()

    #answer,hidden = tagger(batch)
    POSoutput,chunkoutput = tagger(batch)
    answer,hidden = POSoutput

    answerdata = torch.max(answer,2)[1].view(batch.postags.size()).data
    for x,y in zip(batch.sentence.transpose(0,1),answerdata.transpose(0,1)):
        sentence = [inputs.vocab.itos[z] for z in x.data]
        postags = [answers.vocab.itos[z] for z in y]
        for u,v in zip(sentence,postags):
            if u == v == '<pad>':continue
            print('\t::'+u+'::'+v+'::')
        input('... press enter to continue ...')


def progress(b,blen,accu,astr = '[{0}{1}] {2:.2f}% accuracy: {3:.2f}',alen = 40):
    '''Provide a progress bar on stdout between batches'''
    perc = 100.0*b/blen
    anum = ((alen*(b+1))//100)
    print(astr.format('#'*anum,' '*(alen-anum),perc,accu),end='\r')


'''For counting correct answers'''
ncorrect = lambda a,c : (adata(a,c.size()) == c.data).sum()
'''For reaching answer indices'''
adata = lambda a,s : torch.max(a,2)[1].view(s).data
'''For summing losses across sentences'''
sloss = lambda c,a,b : sum([c(a[:,i],b[:,i]) for i in range(a.size()[1])])
'''For printing of examples'''
flatten = lambda seqs : [item for seq in seqs for item in seq]
spair = lambda o,c : (adata(o[0],c.size())[:,0],c.data[:,0])
sline = lambda a,b,c,d : '\t::'+itos(0,a)+'::'+itos(0,b)+'::'+itos(1,c)+'::'+itos(1,d)+'::'
sprint = lambda seq : '\n'.join([sline(*p) for p in zip(*seq)])


if __name__ == '__main__':
    config = util.gather()

    inputs,answers,train_iter,test_iter = fields()

    itos = lambda j,x : answers[j].vocab.itos[x].center(8)

    config.n_embed = len(inputs.vocab)
    config.d_outs = [len(a.vocab) for a in answers]
    tagger = newmodel(inputs.vocab.vectors)

    if config.epochs:train(tagger,train_iter,test_iter)
    
    if config.wiki:
        raise NotImplementedError('MultiTagger implementation temporarily broke work()')
        work(tagger,inputs,answers)


