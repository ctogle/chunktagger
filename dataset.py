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
    text_fields = ('Sentence',)
    tag_fields = ('POStags','Chunks')


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


    @staticmethod
    def jsonline(data):
        line = {'Sentence':data[0],'POStags':data[1],'Chunks':data[2]}
        return line


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
                field_set = set(POSTags.text_fields+POSTags.tag_fields)
                with open(jpath,'w') as jh:
                    for data in zdata:
                        line = cls.jsonline(data)
                        assert set(line.keys()) == field_set
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


