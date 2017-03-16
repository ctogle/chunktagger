import torch
import argparse,functools,os,time


'''For accessing answer indices'''
adata = lambda a,s : torch.max(a,2)[1].view(s).data
'''For counting correct answers'''
ncorrect = lambda a,c : (adata(a,c.size()) == c.data).sum()


def get_progress_function(tasks,barlength = 20):
    '''Provide a function for a progress bar on 
    stdout between batches and associated header.'''
    barlength -= 6
    assert barlength > 1
    mean = lambda l : sum(l)/len(l) if l else 0
    mullist = lambda u,y : [u*v for v in y]
    redlist = lambda l : functools.reduce(lambda x,y : x+y,l)
    accu = lambda j : '{{{0}{1}}}%'.format(j+6,':'+str(len(tasks[j])+2)+'.2f')
    prog_string = '\r {0:3d}  [{1}{2}{3:5.1f}%] {4:8.1f}s {5:9.2f}%'
    accu_string = [accu(j) for j in range(len(tasks))]
    prog_string += redlist(accu_string)

    def progress(epoch,batchnum,batchcnt,corrects,total,stime):
        percent = 100.0*(batchnum+1)/batchcnt
        elapsed = time.time()-stime
        denom = 0 if total == 0 else 100.0/total
        accuracies = mullist(denom,[mean(corrects)]+corrects)
        n = int(percent*barlength//100)
        fmat = [epoch+1,'|'*n,'-'*(barlength-n),percent,elapsed]+accuracies
        print(prog_string.format(*fmat),end = '\r' if percent < 100 else '\n')

    prog_header = 'Epoch '+' '*(barlength+11)+'Elapsed   Accuracy'
    accu_header = ['   {0}'.format(tasks[j]) for j in range(len(tasks))]
    prog_header += redlist(accu_header)
    return prog_header,progress


def gather():
    '''Provide namespace of all run options specified.'''
    cachedir = os.path.join(os.getcwd(),'.cache')
    vectorcache = os.path.join(cachedir,'input_vectors.%s.pt')
    modelcache = os.path.join(cachedir,'model_snapshot.pt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cachedir',type = str,default = cachedir)
    parser.add_argument('--vectorcache',type = str,default = vectorcache)
    parser.add_argument('--modelcache',type = str,default = modelcache)
    parser.add_argument('--trainset',type = str,default = 'train.txt.gz')
    parser.add_argument('--testset',type = str,default = 'test.txt.gz')
    parser.add_argument('--fresh',action = 'store_true')
    parser.add_argument('--lower',action = 'store_true')
    parser.add_argument('--word_vectors',type = str,default = 'glove.42B')
    parser.add_argument('--d_embed',type = int,default = 300)
    parser.add_argument('--rnn',type = str,default = 'GRU')
    parser.add_argument('--d_hidden',type = int,default = 100)
    parser.add_argument('--n_layers',type = int,default = 5)
    parser.add_argument('--emb_dp_ratio',type = float,default = 0.2)
    parser.add_argument('--rnn_dp_ratio',type = float,default = 0.2)
    parser.add_argument('--birnn',action = 'store_true')
    parser.add_argument('--epochs',type = int,default = 100)
    parser.add_argument('--timeout',type = int,default = 10000)
    parser.add_argument('--batch_size',type = int,default = 128)
    parser.add_argument('--optimizer',type = str,default = 'RMSprop')
    parser.add_argument('--lr',type = float,default = 0.0005)
    parser.add_argument('--gpu',type = int,default = -1)
    parser.add_argument('--wiki',action = 'store_true')
    config = parser.parse_args()
    config.vectorcache = config.vectorcache % config.word_vectors
    return config


