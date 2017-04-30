#!/home/cogle/anaconda3/bin/python3.6
import main,dataset
import re,pdb


'''
need to handle some specific patterns:
    
    sentence without a VP indicates some mistagging
        can a corrective retagging occur when ambiguity arises?

    a VP without a preceding NP likely corresponds to the following NP?

    sequences of NPs joined by commas and CCs (either an "AND" or "OR" CC)
    "AND" -> a set of distinct statements all of which are true
    "OR" -> a set of somewhat dependent statements at least one of which is true

    sequences of (NP+VP)s separated by commas and CCs may produce 
    sets of distinct statements
    "AND" -> a set of distinct statements all of which are true
    "OR" -> a set of somewhat dependent statements at least one of which is true

    a NP following a PP modifies the preceding statement

    a VP followed or preceded by an ADJP modifieds a statement

    would be very useful to tag the tense of VPs

    would be very useful perform anaphorism resolution between sentences

    ultimately would like a set of concise, independent statements to analyze
    each such statement should have a specific set of components:
        a primary VP corresponding to a primary NP
        a set of modifying phrases (NPs, ADJPs, PPs, ADVPs) 
            describing the primary NP or VP
        ADJPs modify NPs, ADVPs modify VPs, (PP+NP)s modifying (NP+VP)s

    variants of sentences derived by changing the order of reduction operations
    should not yield contradictory statements 
        (or statements should universally entail one another?)

    breaking of sentences yields a tree of break operations
'''


def printo(o,ks):
    '''Given the output of the tagger on a sentence (o), 
    print in a readable fashion the part of speech tagging and chunking.'''
    print('::'.join([k.center(16) for k in ks]))
    print('::'.join(['-'*16]*3))
    for j in range(len(o[ks[0]])):
        l = [o[k][j].center(16) for k in ks]
        print('::'.join(l))
    for phrase in dataset.POSTags.chunk([o[k] for k in ks]):
        l = [' '.join(phrase[0]).ljust(32),','.join(phrase[2]).rjust(16)]
        print('::'.join(l))


def resolvechunks(o,ks):
    '''Given the output of the tagger on a sentence (o), 
    produce chunk tuples (phrase type, phrase words, phrase parts of speech).'''
    chunks = []
    for phrase in dataset.POSTags.chunk([o[k] for k in ks]):
        phrasetags = set([x if x == 'O' else x[2:] for x in phrase[2]])
        if len(phrasetags) == 1:phrasetag = phrasetags.pop()
        else:
            print('phrasetag ambiguity %s' % str(phrasetags))
            pdb.set_trace()
        chunks.append((phrasetag,tuple(zip(phrase[0],phrase[1]))))
    return chunks


def scanVP(chunks,d = -1):
    #printc(chunks)
    clen = len(chunks)
    if d < 0:
        j = clen
        while j > 0:
            j -= 1
            if chunks[j][0] == 'VP':
                return j
        return 0
    elif d > 0:
        j = 0
        while j < clen:
            if chunks[j][0] == 'VP':
                return j
            j += 1
        return clen-1


def findbreaks(chunks):
    '''Return, if possible, an alternative set of sentences based on tagging.'''

    '''Identify each acceptable break operation on a subvertex (chunked sentence).
    chunks likely fits a known template about which a breakable can be predicted.
    '''
    chunkf = lambda c : (c[0],)+tuple(' '.join(x) for x in tuple(zip(*c[1])))
    printf = lambda c : ('[ %s | %s | %s ]' % chunkf(c))
    printc = lambda c : print(' -> '.join([printf(p) for p in c]))
    
    prints = lambda c : print(' '.join([chunkf(x)[1] for x in c]))

    printc(chunks)
    
    CC_and = ('O',(('and','CC'),))
    if CC_and in chunks:
        bp = chunks.index(CC_and)
        l,r = chunks[:bp],chunks[bp+1:]
        bpl = scanVP(l,-1)
        bpr = scanVP(r,+1)
        aset = [x for x in l[bpl:]+r[:bpr] if not x == ('O',((',',','),))]
        '''each member of aset presents a candidate sentence...'''
        cands = [l[:bpl]+[aseti]+r[bpr:] for aseti in aset]
        for c in cands:prints(c)
        '''some number of comma separated NPs (VP should adjust for tense)'''
        '''a pair of (NP+VP)s'''

        pdb.set_trace()


def breaksentence(o,ks):
    '''Break a sentence into a set of sentences, providing a resolution tree.
      - Each vertex in the tree represents a set of sentences.
      - Each sentence at a vertex represents a subvertex.
      - Each edge in the tree represents a potential break operation
          from a subvertex at a single vertex to another vertex.
      - The set of terminal vertices once break options are exhausted 
          represents the possible set of sentences (subvertices) to return.

    NOTE: Require a function that returns break options (edges) given a sentence.
    '''

    chunks = resolvechunks(o,ks)

    '''
    graph ([list of vertices],[list of edges])
    each vertex is a list of subvertices [list of chunk variants]
    each edge indicates a starting vertex, subvertex, and ending vertex
    '''
    graph = ([[chunks]],[])
    unfinished = [(0,[0])]
    while True:
        unfin = unfinished.pop()
        vertex = unfin[0]
        for subvertex in unfin[1]:
            breakable = findbreaks(graph[0][vertex][subvertex])
            if breakable:

                pdb.set_trace()

            else:
                pdb.set_trace()

        if not unfinished:break

    pdb.set_trace()


def resolvecorpus(o,ks):
    '''Given a list of tagger outputs for a list of sentences, 
    produce a list of independent and more concise sentences 
    representing the same global information.'''

    '''Order of resolution operations:
    1: Anaphorism resolution (removal of pronouns and the like as possible).
    2: Breaking sentences based on CC usage. 
    '''

    printo(o[0],ks)

    o = [breaksentence(s,ks) for s in o]

    #chunks = [resolvechunks(s,ks) for s in o]
    #for chunked in chunks:
    #    chunkstring = ' -> '.join([('[%s | %s | %s]' % chunk) for chunk in chunked])
    #    print(chunkstring+'\n\n')

    pdb.set_trace()


if __name__ == '__main__':
    d = [
        'An elephant, a donkey, and a penguin walk into a bar.',
        'One of them spoke to the bartender, and the others remained silent.',
            ]

    tagger = main.main() 
    ks = (tagger.config.target_field,)+tagger.config.output_fields

    o = [tagger.work(s) for s in d]
    resolvecorpus(o,ks)

    pdb.set_trace()


