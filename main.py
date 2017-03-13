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
    util.translations(inputs,answers)
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
    loss = sum([util.sloss(crit,o[0],c) for o,c in zip(tdata,bdata)])
    loss.backward();opt.step()
    if v:print(util.sprint(util.flatten([util.spair(o,c) for o,c in zip(tdata,bdata)])))
    return sum([util.ncorrect(o[0],c) for o,c in zip(tdata,bdata)])


def train_epoch(tagger,crit,opt,batcher,epoch,stime):
    '''Perform a training epoch given an iterator of training batches,
    returning the accuracy of the model on the data set.'''
    progress(epoch,-1,len(batcher),-1,stime)
    batcher.init_epoch()
    correct,total = 0,0
    for j,batch in enumerate(batcher):
        bdata = batch.postags,batch.chunks
        correct += train_batch(tagger,crit,opt,batch,j == 0 and config.print_example)
        total += sum([batch.batch_size*b.size()[0] for b in bdata])
        progress(epoch,j,len(batcher),100.0*correct/total,stime)
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
    improvement_threshold = 0.0
    print('... training model ...')
    print(prog_header)
    for j in range(config.epochs):
        try:
            stime = time.time()
            accuracy = train_epoch(tagger,criterion,opt,train_batcher,j,stime)
            improvement = accuracy-lastaccuracy
            lastaccuracy = accuracy
            if improvement < improvement_threshold:
                print('... improvement is quite low ... ending training ...')
                break
            elif improvement > 0.0:torch.save(tagger,config.modelcache)
        except KeyboardInterrupt:
            progress(j,0,1,lastaccuracy if j > 0 else -1,stime)
            print('... training forcefully exited ...')
            torch.save(tagger,config.modelcache)
            break
    print('... testing model ...')
    print(prog_header)
    accuracy = test(tagger,test_batcher)
    print('... final model accuracy: %.2f ...' % accuracy)


def test_batch(tagger,batch,v = False):
    '''Perform testing on a single batch of test examples,
    returning the number of correct answers.'''
    tagger.eval()
    bdata = batch.postags,batch.chunks
    tdata = tagger(batch)
    if v:print(util.sprint(util.flatten([util.spair(o,c) for o,c in zip(tdata,bdata)])))
    return sum([util.ncorrect(o[0],c) for o,c in zip(tdata,bdata)])


def test(tagger,batcher):
    '''Perform testing given an iterator of testing batches,
    returning the accuracy of the model on the data set.'''
    stime = time.time()
    progress(0,-1,len(batcher),-1,stime)
    tagger.eval();batcher.init_epoch()
    correct,total = 0,0
    for j,batch in enumerate(batcher):
        bdata = batch.postags,batch.chunks
        correct += test_batch(tagger,batch,j == 0 and config.print_example)
        total += sum([batch.batch_size*b.size()[0] for b in bdata])
        progress(0,j,len(batcher),100.0*correct/total,stime)
    return 100.0*correct/total


def work(tagger,inputs,answers):
    '''As an example of totally distinct data usage, use the WikiData class 
    to iterate over tagged wiki sentences, printing the chunk phrases.'''
    print('... tag results for wiki data sentence by sentence ...')
    for sentence in dataset.WikiData.gen(tagger,inputs,answers):
        for phrase in dataset.WikiData.chunk(sentence):
            print(' '.join(phrase[0]),'\t',','.join(phrase[2]))
        input('... press enter to continue ...')


prog_header = 'Epoch '+' '*40+'   Complete  Elapsed  Accuracy'
prog_string = '\r {0:3d}  [{1}{2}]   {3:5.1f}%  {4:6.1f}s   {5:6.2f}%'
def progress(e,b,blen,accu,stime,astr = prog_string,alen = 40):
    '''Provide a progress bar on stdout between batches'''
    p = 100.0*(b+1)/blen
    n = int(p*alen//100)
    t = time.time()-stime
    echar = '\r' if p < 100 else '\n'
    print(prog_string.format(e+1,'|'*n,'-'*(alen-n),p,t,accu),end = echar)


if __name__ == '__main__':
    config = util.gather()

    inputs,answers,train_iter,test_iter = fields()

    config.n_embed = len(inputs.vocab)
    config.d_outs = [len(a.vocab) for a in answers]
    tagger = newmodel(inputs.vocab.vectors)

    if config.epochs:train(tagger,train_iter,test_iter)
    
    if config.wiki:work(tagger,inputs,answers)


