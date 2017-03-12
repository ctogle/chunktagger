import torchtext
import pdb,os,urllib,json,math

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
            zdata = util.unpack(zpath)
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
    def splits(cls,text_field,label_field,root = '.',
            train = 'train.txt.json',validation = None,test = 'test.txt.json'):
        path = cls.download_or_unzip(root)
        return super(POSTags,cls).splits(
            os.path.join(path,''),train,validation,test,
            format = 'json',fields = {
                'sentence': ('sentence',text_field),
                'postags': ('postags',label_field),
                #'pos': ('poss',label_field),
                #'chunk': ('chunks',label_field),
                    })
            #        },
            #filter_pred = lambda ex: ex.label != '-')


