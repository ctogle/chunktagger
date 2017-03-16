#!/home/cogle/anaconda3/bin/python3.6
import torch,torchtext
import os,time,pdb
import util,dataset,model


def fields(dclass = dataset.POSTags):
    '''Create field objects and train and test data sets.
    Use the datasets to initialize the vocab objects of the fields.'''
    inputs = torchtext.data.Field(lower = config.lower)
    answers = [torchtext.data.Field() for x in range(len(dclass.tag_fields))]
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
    target_field = dataset.POSTags.text_fields[0]
    kws = {
        'batch_size' : config.batch_size,
        'device' : config.gpu,
        'sort_key' : lambda x : len(x.__getattribute__(target_field)),
        'repeat' : False,
            }
    train_i,test_i = torchtext.data.BucketIterator.splits(dsets,**kws)
    return target_field,inputs,answers,train_i,test_i


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


'''For summing losses across sentences'''
sloss = lambda c,a,b : sum([c(a[:,i],b[:,i]) for i in range(a.size()[1])])
def train_batch(tagger,crit,opt,batch,bdataf):
    '''Perform training on a single batch of examples, 
    returning the number of correct answers'''
    tagger.train();opt.zero_grad()
    tdata,bdata = tagger(batch),bdataf(batch)
    loss = sum([sloss(crit,o[0],c) for o,c in zip(tdata,bdata)])
    loss.backward();opt.step()
    return [util.ncorrect(o[0],c) for o,c in zip(tdata,bdata)]


def train_epoch(tagger,crit,opt,batcher,epoch,stime,bdataf,progf):
    '''Perform a training epoch given an iterator of training batches,
    returning the accuracy of the model on the data set.'''
    progf(epoch,-1,len(batcher),[-1]*config.n_taggers,0,stime)
    batcher.init_epoch()
    corrects,total = [0]*config.n_taggers,0
    for j,batch in enumerate(batcher):
        newcorrects = train_batch(tagger,crit,opt,batch,bdataf)
        corrects = [x+y for x,y in zip(corrects,newcorrects)]
        total += batch.batch_size*bdataf(batch)[0].size()[0]
        progf(epoch,j,len(batcher),corrects,total,stime)
        if time.time()-stime > config.timeout:raise KeyboardInterrupt
    return 100.0*sum(corrects)/(total*config.n_taggers)


def train(tagger,train_i,test_i,bdataf,prog_header,progf):
    '''Perform training of the model given an iterator of training batches.
    Exit the training process early on KeyboardInterrupt, or if accuarcy 
    improvement is sufficiently slow.
    Save the model between training epochs or upon early exit.
    Test the accuracy of the model on an iterator of test batches when 
    training is complete.'''
    lossf = torch.nn.CrossEntropyLoss()
    if hasattr(torch.optim,config.optimizer):
        optclass = torch.optim.__getattribute__(config.optimizer)
        opt = optclass(tagger.parameters(),lr = config.lr)
    else:raise ValueError('... unavailable optimizer: %s ...' % config.optimizer)
    lastaccuracy = 0.0
    improvement_threshold = 0.0
    print('... training model ...')
    print(prog_header)
    stime = time.time()
    for j in range(config.epochs):
        try:
            accuracy = train_epoch(tagger,lossf,opt,train_i,j,stime,bdataf,progf)
            improvement = accuracy-lastaccuracy
            lastaccuracy = accuracy
            if improvement < improvement_threshold:
                print('... improvement is quite low ... ending training ...')
                break
            elif improvement > 0.0:torch.save(tagger,config.modelcache)
        except KeyboardInterrupt:
            progf(j,0,1,[-1]*config.n_taggers,0,stime)
            print('... training forcefully exited ...')
            torch.save(tagger,config.modelcache)
            break
    print('... testing model ...')
    print(prog_header)
    accuracy = test(tagger,test_i,bdataf,progf)
    print('... final model task-averaged accuracy: %.2f ...' % accuracy)


def test_batch(tagger,batch,bdataf):
    '''Perform testing on a single batch of test examples,
    returning the number of correct answers.'''
    tagger.eval()
    tdata,bdata = tagger(batch),bdataf(batch)
    return [util.ncorrect(o[0],c) for o,c in zip(tdata,bdata)]


def test(tagger,batcher,bdataf,progf):
    '''Perform testing given an iterator of testing batches,
    returning the accuracy of the model on the data set.'''
    stime = time.time()
    progf(0,-1,len(batcher),[-1]*config.n_taggers,0,stime)
    tagger.eval();batcher.init_epoch()
    corrects,total = [0]*config.n_taggers,0
    for j,batch in enumerate(batcher):
        newcorrects = test_batch(tagger,batch,bdataf)
        corrects = [x+y for x,y in zip(corrects,newcorrects)]
        total += batch.batch_size*bdataf(batch)[0].size()[0]
        progf(0,j,len(batcher),corrects,total,stime)
    return 100.0*sum(corrects)/(total*config.n_taggers)


def work(config,tagger,inputs,answers):
    '''As an example of totally distinct data usage, use the WikiData class 
    to iterate over tagged wiki sentences, printing the chunk phrases.'''
    print('... tag results for wiki data sentence by sentence ...')
    for sentence in dataset.WikiData.gen(config,tagger,inputs,answers):
        for phrase in dataset.WikiData.chunk(sentence):
            print(' '.join(phrase[0]),'\t',','.join(phrase[2]))
        input('... press enter to continue ...')


def main(config):
    '''Main loop for use creates Fields using training/test data, instantiates 
    a model (loading existing parameters if specified), trains the model if 
    specified, and runs the wiki data work example if specified.'''
    target_field,inputs,answers,train_i,test_i = fields()

    fkeys = train_i.dataset.fields.keys()
    bkeys = [k for k in fkeys if train_i.dataset.fields[k] in answers]
    bdataf = lambda b : tuple(b.__getattribute__(s) for s in bkeys)

    config.target_field = target_field
    config.n_taggers = len(bkeys)
    config.n_embed = len(inputs.vocab)
    config.d_hidden = [config.d_hidden]*len(answers)
    config.d_out = [len(a.vocab) for a in answers]

    tagger = newmodel(inputs.vocab.vectors)
    prog_head,prog_func = util.get_progress_function(bkeys)
    if config.epochs:train(tagger,train_i,test_i,bdataf,prog_head,prog_func)
    if config.wiki:work(config,tagger,inputs,answers)


if __name__ == '__main__':
    config = util.gather()
    main(config)


