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
    dirname = 'POSTag_data'


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


    @classmethod
    def download_or_unzip(cls,root):
        path = os.path.join(root,cls.dirname)
        if not os.path.isdir(path):os.mkdir(path)
        for url,filename in zip(cls.urls,cls.filenames):
            zpath = os.path.join(path,filename)
            if os.path.isfile(zpath):
                print('zpath already exists: %s' % zpath)
            else:
                print('downloading: %s' % url)
                urllib.request.urlretrieve(url,zpath)
                print('downloaded: %s' % url)
            zdata = POSTags.unpack(zpath)
            jpath = zpath[:zpath.rfind('.')]+'.json'
            # may skip the writing if jpath exists...
            with open(jpath,'w') as jh:
                for data in zdata:
                    line = {
                        'sentence':data[0],
                        'postags':data[1],
                        'chunks':data[2],
                            }
                    json.dump(line,jh)
                    jh.write(os.linesep)
        return path


    @classmethod
    def splits(cls,text_field,label_fields,root = '.',
            train = 'train.txt.json',validation = None,test = 'test.txt.json'):
        path = cls.download_or_unzip(root)
        return super(POSTags,cls).splits(
            os.path.join(path,''),train,validation,test,
            format = 'json',fields = {
                'sentence': ('sentence',text_field),
                'postags': ('postags',label_fields[0]),
                'chunks': ('chunks',label_fields[1]),
                    })
            #        },
            #filter_pred = lambda ex: ex.label != '-')


class WikiData(POSTags):


    urls = ()
    filenames = ()
    dirname = 'Wiki_data'
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
        # may skip the writing if jpath exists...
        with open(jpath,'w') as jh:
            for q in documents:
                d = documents[q] 
                for s in d:
                    line = {
                        'sentence':s,
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


