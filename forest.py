import torch,torchtext
import os,random,functools,itertools,copy,time,pdb
import model,util


class Forest(object):


    treename = 'forest.%d.pt'


    def addtree(self,variant,data,treename = None):
        treecnt = len(self.trees)+1
        print('... forest will have %d MultiTagger "trees" ...' % treecnt)
        c = copy.deepcopy(self.config)
        if treename is None:treename = self.treename % (len(self.trees)+1)
        c.modelcache = os.path.join(c.cachedir,treename)
        c.d_hidden,c.n_layers,c.rnn = variant
        c.targetaccuracy = self.config.targetaccuracy
        c.epochs = self.config.epochs
        tree = model.newmodel(c,data)
        self.trees.append(tree)
        return tree


    def pickvariant(self):
        d_hidden = [5*random.randint(2,20) for x in range(2)]
        n_layers = random.randint(1,3)
        #rnn = random.choice(('GRU','LSTM'))
        rnn = 'GRU'
        return d_hidden,n_layers,rnn


    def whichleastaccurate(self):
        which = 0
        threshold = self.trees[which].config.testing_accuracy
        for j in range(len(self.trees)):
            if self.trees[j].config.testing_accuracy < threshold:
                which = j
                threshold = self.trees[which].config.testing_accuracy
        return which


    def whichleastdiverse(self,data):
        train_i,test_i = data['dataset_iters']
        inputs,answers = data['fields']

        test_i.init_epoch()

        e_sq_d = lambda x,y : torch.exp(-(x-y)**2)

        for j,batch in enumerate(test_i):
            i = batch.__getattribute__(tagger.config.target_field)
            o = self(i)

            '''
            goal: pick the model which yields the most similar results
            as all other models on the test data

            so, e_sq_d between each pairwise combo of models and for each
            task should be minimized for the selection
            '''
            similarities = []
            for d in o:
                #sim = functools.reduce(e_sq_d,d)
                #torch.FloatTensor([e_sq_d(x,y) for x in d for y in d])
                #pairs = [[(x,y) for x in d] for y in d]
                simsize = (len(d),)*2+d[0].size()
                simt = torch.FloatTensor(*simsize)
                for j,x in enumerate(d):
                    for k,y in enumerate(d):
                        simt[j,k,:,:,:] = e_sq_d(x,y).data
                similarities.append(simt)

            pdb.set_trace()

        '''#
        for d in tdatas:
            #sim = functools.reduce(e_sq_d,d)
            #torch.FloatTensor([e_sq_d(x,y) for x in d for y in d])
            #pairs = [[(x,y) for x in d] for y in d]
            simsize = (len(d),)*2+d[0].size()
            simt = torch.FloatTensor(*simsize)
            for j,x in enumerate(d):
                for k,y in enumerate(d):
                    simt[j,k,:,:,:] = e_sq_d(x,y).data
            print([simt[0,j].mean() for j in range(20)])
            print([simt[j,0].mean() for j in range(20)])
            print(simt[0,:].mean())
            print(simt[:,0].mean())
            print('for example')
            print(['%.5f' % v for v in simt[0,1,0,0,:]])
            pdb.set_trace()
        '''#

        return which


    def prunegrow(self,data):
        which = self.whichleastaccurate()
        #which = self.whichleastdiverse(data)
        pruned = self.trees.pop(which)
        os.remove(pruned.config.modelcache)
        print('... pruning tree %s ...' % pruned.config.modelcache)
        pruned.report_architecture()
        grown = self.addtree(self.pickvariant(),data,pruned.config.modelcache)
        print('... grew tree %s ...' % grown.config.modelcache)
        return self


    def __iter__(self):
        return self.trees.__iter__()


    def __init__(self,c):
        self.config = c
        self.trees = []
        self.test_accuracy = -1.0
        self.legacy = []


    def __repr__(self):
        forestacc = [t.config.testing_accuracy for t in self]
        forestmin = min(forestacc)
        forestmax = max(forestacc)
        forestavg = sum(forestacc)/len(forestacc)
        concensus = self.testing_accuracy-forestmax
        s = '-'*40
        s += '\n... individual testing minimum %.2f ...' % forestmin
        s += '\n... individual testing maximum %.2f ...' % forestmax
        s += '\n... individual testing average %.2f ...' % forestavg
        s += '\n... concensus testing accuracy %.2f ...' % self.testing_accuracy
        s += '\n... concensus testing success %.2f ...' % concensus
        legacystr = ','.join([str(v) for v in self.legacy])
        s += '\n... legacy values: %s ...' % legacystr
        s += '\n'+'-'*40
        return s


    def __call__(self,i):
        o = [t(i) for t in self]
        o = list(zip(*[[d[k][0] for k in range(2)] for d in o]))
        return o


    def resolve(self,answers):
        #tdatas = [sum(d) for d in answers]
        #tdatas = [torch.max(d,2)[1] for d in tdatas]
        #return tdatas

        tdatas = [torch.cat([torch.max(d,2)[1] for d in a],2) for a in answers]
        #tdatas = [d.mean(2)[0][:,:,0] for d in tdatas]
        tdatas = [d.mode(2)[0][:,:,0] for d in tdatas]
        return tdatas 


    def test(self,data):
        train_i,test_i = data['dataset_iters']
        inputs,answers = data['fields']
        fkeys = test_i.dataset.fields.keys()
        bkeys = [k for k in fkeys if test_i.dataset.fields[k] in answers]
        bdataf = lambda b : tuple(b.__getattribute__(s) for s in bkeys)
        prog_header,progf = util.get_progress_function(bkeys)
        if progf:print(prog_header)
        n_taggers = self.trees[0].config.n_taggers
        stime = time.time()
        progf(0,-1,len(test_i),[-1]*n_taggers,0,stime)
        test_i.init_epoch()
        corrects,total = [0]*n_taggers,0
        for j,batch in enumerate(test_i):
            b = bdataf(batch)
            i = batch.__getattribute__(self.trees[0].config.target_field)
            o = self.resolve(self(i))
            newcorrects = [(d.data == c.data).sum() for d,c in zip(o,b)]
            corrects = [x+y for x,y in zip(corrects,newcorrects)]
            total += batch.batch_size*b[0].size()[0]
            progf(0,j,len(test_i),corrects,total,stime)
        concensus_accuracy = 100.0*sum(corrects)/(total*n_taggers)
        self.testing_accuracy = concensus_accuracy
        self.legacy.append(self.testing_accuracy)
        return concensus_accuracy


def newforest(config,data):

    '''
    What differs between instances of the MultiTagger models?
        -   modelcache
    What hyperparameters differ between relevant architectures?
        -   d_hidden
        -   n_layers
        -   rnn
    '''

    '''
    variations = list(itertools.product(*(
        #([10,90],[20,80],[30,70],[40,60],[50,50],
        #    [60,40],[70,30],[80,20],[90,10]),
        ([20,10]*20),
        #([25,75],[50,50],[75,25],[50,100],[75,75],[100,50]),
        (2,),
        ('GRU',),
        #('GRU','LSTM'),
            )))
    for variant in variations:
        forest.addtree(variant,data)
    '''

    variations = 50

    forest = Forest(config)
    for variant in range(variations):
        forest.addtree(forest.pickvariant(),data)
    forest.test(data)
    print(forest)

    while True:
        try:
            forest.prunegrow(data)
            forest.test(data)
            print(forest)
        except KeyboardInterrupt:
            print('... pruning and growing ended ...')
            break

    '''
    forestacc = [t.config.testing_accuracy for t in forest]
    forestmin = min(forestacc)
    forestmax = max(forestacc)
    forestavg = sum(forestacc)/len(forestacc)
    print('... individual testing minimum ...',forestmin)
    print('... individual testing maximum ...',forestmax)
    print('... individual testing average ...',forestavg)

    concensus_accuracy = forest.test(data)
    print('... concensus testing ...',concensus_accuracy)
    print('... success: %f ...' % (concensus_accuracy-forestmax))
    '''

    '''
    using modal concensus seems to offer an advantage

    variability within the forest current comes from uniformly different
    randomized weight values, and distinct architecture differences

    what other concensus schemes lead to an advantage?
    does the advantage depend on the presence of overtraining?
    which of these is really driving the advantage though?
    does the advantage increase with additional forest members?
    does the advantage depend on which architecture parameters are distinct?
    could a grow and prune scheme be introduced that explores the 
        space of possible architectures, pruning for accuracy?

    need to test advantage with uniform architecture but variable forest size
    need to test advantage with uniform architecture but deeper weight 
        randomization variability
    '''

    return forest





