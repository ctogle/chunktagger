import torchtext
import wikipedia,os,re,urllib,json,gzip,pdb
import util


class POSTags(torchtext.data.TabularDataset):


    urls = (
        'http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz',
        'http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz',
            )
    filenames = (
        'test.txt.gz',
        'train.txt.gz',
            )
    dirname = '.POSTag_data'
    text_fields = ('sentence','reversal')
    tag_fields = ('postags','chunks')


    @staticmethod
    def unpack(path):
        '''Generator for sentences found in train and test data gzip files.'''
        with gzip.open(path,'rb') as f:
            content = f.read().decode('unicode_escape')
        s = []
        for l in content.split(os.linesep):
            if l:s.append(tuple(l.split(' ')))
            elif s:
                yield tuple(zip(*s))
                s = []


    @staticmethod
    def chunk(sentence):
        '''Generator for sentence chunks where sentence is a sequence 
        of sequences of the form (word, POS tag, chunk tag).'''
        zipped = zip(*sentence)
        piece = [next(zipped)]
        for item in zipped:
            chunktag = item[2]
            if [chunktag.startswith(i) for i in ('<pad>','B','O')].count(True):
                yield tuple(zip(*piece))
                piece = [item]
            elif chunktag.startswith('I'):
                #assert chunktag[1:] == piece[-1][2][1:]
                piece.append(item)
            else:
                pdb.set_trace()
                raise ValueError('unknown chunk tag: %s' % chunktag)
        yield tuple(zip(*piece))


    @classmethod
    def download_or_unzip(cls,root):
        path = os.path.join(root,cls.dirname)
        if not os.path.isdir(path):os.mkdir(path)
        for url,filename in zip(cls.urls,cls.filenames):
            zpath = os.path.join(path,filename)
            if not os.path.isfile(zpath):
                print('... downloading: %s ...' % url)
                urllib.request.urlretrieve(url,zpath)
                print('... downloaded: %s ...' % url)
            zdata = POSTags.unpack(zpath)
            jpath = zpath[:zpath.rfind('.')]+'.json'
            if not os.path.exists(jpath):
                with open(jpath,'w') as jh:
                    field_set = set(cls.text_fields+cls.tag_fields)
                    for data in zdata:
                        line = {
                            'sentence':data[0],
                            'reversal':data[0][::-1],
                            'postags':data[1],
                            'chunks':data[2],
                                }
                        assert set(lines.keys()) == field_set
                        json.dump(line,jh)
                        jh.write(os.linesep)
        return path


    @classmethod
    def splits(cls,text_field,label_fields,root = '.',
            train = 'train.txt.json',validation = None,test = 'test.txt.json'):
        path = cls.download_or_unzip(root)
        fields = dict(
            [(k,(k,text_field)) for k in cls.text_fields]+\
            [(k,(k,label_fields[j])) for j,k in enumerate(cls.tag_fields)])
        return super(POSTags,cls).splits(
            os.path.join(path,''),train,validation,test,
            format = 'json',fields = fields)


class WikiData(POSTags):


    urls = ()
    filenames = ()
    dirname = '.Wiki_data'
    queries = (
        'Python (programming language)',
            )


    @staticmethod
    def breakparagraph(paragraph):
        end = re.compile('[.!?]')
        sentences = []
        sentence = ''
        for word in paragraph.split(' '):
            if end.findall(word):
                sentence += ' '+word
                sentence = sentence.replace('.',' .')
                sentence = sentence.replace(',',' ,')
                sentence = sentence.replace('\'',' \'')
                if os.linesep in sentence:
                    sentence = sentence[sentence.rfind(os.linesep):]
                yield sentence
                sentence = ''
            elif sentence:sentence += ' '+word
            else:sentence += word


    @staticmethod
    def wiki(query):
        '''Query wikipedia and return resulting sentences.'''
        sentences = []
        try:
            print('... scouring wikipedia for \'%s\' ...' % query)
            wp = wikipedia.page(query)
            print('... found \'%s\' at URL:\n\t <%s> ...' % (wp.title,wp.url))
            paragraphs = wp.content.split(os.linesep)
            for p in paragraphs:sentences.extend(WikiData.breakparagraph(p))
        except wikipedia.exceptions.DisambiguationError:
            print('... query \'%s\' not found on wikipedia ...' % query)
        return sentences


    @classmethod
    def download_or_unzip(cls,root):
        path = os.path.join(root,cls.dirname)
        if not os.path.isdir(path):os.mkdir(path)
        documents = {}
        for query in cls.queries:
            filepath = os.path.join(path,query+'.wikipage')
            documents[query] = []
            if os.path.exists(filepath):
                with open(filepath,'r') as fh:
                    for sentence in fh.readlines():
                        documents[query].append(sentence.strip().split(' '))
            else:
                sentencegen = WikiData.wiki(query)
                with open(filepath,'w') as fh:
                    for sentence in sentencegen:
                        fh.write(sentence+os.linesep)
                        documents[query].append(sentence.split(' '))
        jpath = os.path.join(path,'wiki.txt.json')
        if not os.path.exists(jpath):
            with open(jpath,'w') as jh:
                for q in documents:
                    d = documents[q] 
                    for s in d:
                        line = {
                            'sentence':s,
                            'reversal':s[::-1],
                            'postags':s,
                            'chunks':s,
                                }
                        json.dump(line,jh)
                        jh.write(os.linesep)
        return path


    @classmethod
    def splits(cls,*args,root = '.',
            train = None,validation = None,test = 'wiki.txt.json'):
        return super(WikiData,cls).splits(*args,root,train,validation,test)


    @staticmethod
    def gen(tagger,inputs,answers):
        '''As an example of totally distinct data usage, create a Dataset of
        Wikipedia page sentences, a single Batch for all of the sentences, 
        and run the model on them. Yield the tag results of each sentence.'''
        dset = WikiData.splits(inputs,answers)[0]
        batch = torchtext.data.Batch(dset.examples,dset,tagger.config.gpu,False)
        tagger.eval()
        bdata = batch.postags,batch.chunks
        tdata = [util.adata(o[0],c.size()) for o,c in zip(tagger(batch),bdata)]
        for i,sentence in enumerate(batch.sentence.transpose(0,1).data):
            words = [[util.iitos(x).strip()] for x in sentence]
            spacer = max([len(w[0]) for w in words])+2
            for j,td in enumerate(tdata):
                tdat = td.transpose(0,1)
                tags = [util.aitos(j,x).strip() for x in tdat[i,:]]
                words = [l+[t] for l,t in zip(words,tags)]
            reconstructed = []
            for w in words:
                if w[0] == '<pad>':break
                reconstructed.append(tuple(w))
                print('::'.join([l.center(spacer) for l in w]))
            yield zip(*reconstructed)


