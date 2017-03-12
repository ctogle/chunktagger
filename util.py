import argparse,os,gzip


def unpack(path):
    with gzip.open(path,'rb') as f:
        content = f.read().decode('unicode_escape')
    s = []
    for l in content.split(os.linesep):
        if l:s.append(tuple(l.split(' ')))
        elif s:
            yield tuple(zip(*s))
            s = []


def chunk(sentence):
    zipped = zip(*sentence)
    piece = [next(zipped)]
    for item in zipped:
        chunktag = item[2]
        if chunktag.startswith('B') or chunktag.startswith('O'):
            yield tuple(zip(*piece))
            piece = [item]
        elif chunktag.startswith('I'):
            assert chunktag[1:] == piece[-1][2][1:]
            piece.append(item)
        else:raise ValueError('unknown chunk tag: %s' % chunktag)
    yield tuple(zip(*piece))
    '''#
    i = 0
    for sentence in trainset:
        for phrase in chunk(sentence):
            print(' '.join(phrase[0]),'\t',','.join(phrase[2]))
        i += 1
        if i == 5:
            pdb.set_trace()
    '''#


def gather():
    cachedir = os.path.join(os.getcwd(),'cache')
    vectorcache = os.path.join(cachedir,'input_vectors.pt')
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
    #parser.add_argument('--d_hidden',type = int,default = 300)
    parser.add_argument('--d_hidden',type = int,default = 100)
    parser.add_argument('--n_layers',type = int,default = 1)
    parser.add_argument('--dp_ratio',type = float,default = 0.2)
    parser.add_argument('--birnn',action = 'store_true')
    parser.add_argument('--epochs',type = int,default = 10)
    parser.add_argument('--batch_size',type = int,default = 128)
    parser.add_argument('--learningrate',type = float,default = 0.001)
    parser.add_argument('--gpu',type = int,default = -1)
    config = parser.parse_args()
    return config


