#!/home/cogle/anaconda3/bin/python3.6
import torchtext
import os,re,wikipedia,json,pdb
import dataset,util,main


class WikiData(dataset.POSTags):


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


    @staticmethod
    def jsonline(data):
        field_set = set(WikiData.text_fields+WikiData.tag_fields)
        line = {'Sentence':data,'POStags':data,'Chunks':data}
        assert set(line.keys()) == field_set
        return line


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
            field_set = set(WikiData.text_fields+WikiData.tag_fields)
            with open(jpath,'w') as jh:
                for q in documents:
                    d = documents[q] 
                    for s in d:
                        line = cls.jsonline(s)
                        assert set(line.keys()) == field_set
                        json.dump(line,jh)
                        jh.write(os.linesep)
        return path


    @classmethod
    def splits(cls,*args,root = '.',
            train = None,validation = None,test = 'wiki.txt.json'):
        return super(WikiData,cls).splits(*args,root,train,validation,test)


    @staticmethod
    def gen(config,tagger,inputs,answers):
        '''As an example of totally distinct data usage, create a Dataset of
        Wikipedia page sentences, a single Batch for all of the sentences, 
        and run the model on them. Yield the tag results of each sentence.'''
        iitos = lambda x : inputs.vocab.itos[x].center(8)
        aitos = lambda j,x : answers[j].vocab.itos[x].center(8)
        dset = WikiData.splits(inputs,answers)[0]
        batch = torchtext.data.Batch(dset.examples,dset,config.gpu,False)
        tagger.eval()
        bdata = [batch.__getattribute__(k) for k in WikiData.tag_fields]
        i = batch.__getattribute__(tagger.config.target_field)
        tdata = [util.adata(o[0],c.size()) for o,c in zip(tagger(i),bdata)]
        for i,sentence in enumerate(batch.Sentence.transpose(0,1).data):
            words = [[iitos(x).strip()] for x in sentence]
            spacer = max([len(w[0]) for w in words])+2
            for j,td in enumerate(tdata):
                tdat = td.transpose(0,1)
                tags = [aitos(j,x).strip() for x in tdat[i,:]]
                words = [l+[t] for l,t in zip(words,tags)]
            reconstructed = []
            for w in words:
                if w[0] == '<pad>':break
                reconstructed.append(tuple(w))
                print('::'.join([l.center(spacer) for l in w]))
            yield zip(*reconstructed)


def work(config,tagger,inputs,answers):
    '''As an example of totally distinct data usage, use the WikiData class 
    to iterate over tagged wiki sentences, printing the chunk phrases.'''
    print('... tag results for wiki data sentence by sentence ...')
    graphs = []
    for sentence in WikiData.gen(config,tagger,inputs,answers):
        graph = [[],[]]

        for phrase in WikiData.chunk(sentence):
            print(' '.join(phrase[0]),'\t',','.join(phrase[2]))

            words,postags,chunktags = phrase
            if phrase == ((',',),(',',),('O',)):
                print('read , ...')
                continue
            if phrase == (('and',),('CC',),('O',)):
                print('read and ...')
                continue
            if phrase == (('or',),('CC',),('O',)):
                # need to handle: "rather than x or y"
                print('read and ...')
                continue
            if phrase == (('.',),('.',),('O',)):
                print('read . ...')
                continue

            try:
                chunktags,phrasetags = zip(*[x.split('-') for x in chunktags])
            except ValueError:
                pdb.set_trace()

            phrasetag = phrasetags[0]

            if phrasetag == 'NP':
                vertex = (words,postags,phrasetag)
                graph[0].append(vertex)
            elif phrasetag == 'VP':
                edge = (words,postags,phrasetag,len(graph[0])-1)
                graph[1].append(edge)
            elif phrasetag == 'PP':
                edge = (words,postags,phrasetag,len(graph[0])-1)
                graph[1].append(edge)
            elif phrasetag == 'ADJP':
                print('ADJP poorly handled')
                edge = (words,postags,phrasetag,len(graph[0])-1)
                graph[1].append(edge)
            elif phrasetag == 'ADVP':
                print('ADVP poorly handled')
                edge = (words,postags,phrasetag,len(graph[0])-1)
                graph[1].append(edge)
            elif phrasetag == 'SBAR':
                print('SBAR poorly handled')
                edge = (words,postags,phrasetag,len(graph[0])-1)
                graph[1].append(edge)
            else:pdb.set_trace()

        #input('... press enter to continue ...')
        graphs.append(graph)

    '''Resolve each graph into a set of simplified, independent sentences.'''
    pdb.set_trace()


if __name__ == '__main__':
    tagger = main.main()
    work(tagger.config,tagger,*tagger.config.fields)


