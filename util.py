import torch
import argparse,os


'''For counting correct answers'''
ncorrect = lambda a,c : (adata(a,c.size()) == c.data).sum()
'''For reaching answer indices'''
adata = lambda a,s : torch.max(a,2)[1].view(s).data
'''For summing losses across sentences'''
sloss = lambda c,a,b : sum([c(a[:,i],b[:,i]) for i in range(a.size()[1])])
'''For printing of examples'''
flatten = lambda seqs : [item for seq in seqs for item in seq]
spair = lambda o,c : (adata(o[0],c.size())[:,0],c.data[:,0])
sline = lambda a,b,c,d : '\t::'+aitos(0,a)+'::'+aitos(0,b)+'::'+aitos(1,c)+'::'+aitos(1,d)+'::'
sprint = lambda seq : '\n'.join([sline(*p) for p in zip(*seq)])


iitos,aitos = None,None
def translations(inputs,answers):
    global iitos,aitos
    iitos = lambda x : inputs.vocab.itos[x].center(8)
    aitos = lambda j,x : answers[j].vocab.itos[x].center(8)


def gather():
    cachedir = os.path.join(os.getcwd(),'cache')
    vectorcache = os.path.join(cachedir,'input_vectors.pt')
    modelcache = os.path.join(cachedir,'model_snapshot.pt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cachedir',type = str,default = cachedir)
    parser.add_argument('--vectorcache',type = str,default = vectorcache)
    parser.add_argument('--modelcache',type = str,default = modelcache)
    parser.add_argument('--n_taggers',type = int,default = 2)
    parser.add_argument('--trainset',type = str,default = 'train.txt.gz')
    parser.add_argument('--testset',type = str,default = 'test.txt.gz')
    parser.add_argument('--fresh',action = 'store_true')
    parser.add_argument('--lower',action = 'store_true')
    parser.add_argument('--word_vectors',type = str,default = 'glove.42B')
    parser.add_argument('--d_embed',type = int,default = 300)
    parser.add_argument('--d_hidden',type = int,default = 100)
    parser.add_argument('--n_layers',type = int,default = 1)
    parser.add_argument('--dp_ratio',type = float,default = 0.2)
    parser.add_argument('--birnn',action = 'store_true')
    parser.add_argument('--epochs',type = int,default = 10)
    parser.add_argument('--batch_size',type = int,default = 128)
    parser.add_argument('--learningrate',type = float,default = 0.001)
    parser.add_argument('--gpu',type = int,default = -1)
    parser.add_argument('--print_example',action = 'store_true')
    parser.add_argument('--wiki',action = 'store_true')
    config = parser.parse_args()
    return config


