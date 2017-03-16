# chunktagger
Part-of-speech tagging/chunking using recurrent neural networks.

chunktagger provides a script (main.py) for creating, training, and using a recurrent neural network model 
(primarily LSTM or GRU module provided by pytorch) to perform part-of-speech tagging and chunking on sentences.
It was developed using the Anaconda Python 3 distribution with pytorch and torchtext installed as well.

For usage, run:
  python main.py -h

The training and test data are from the Wall Street Journal corpus, and are found at two URLs:
  http://www.cnts.ua.ac.be/conll2000/chunking/train.txt.gz
  http://www.cnts.ua.ac.be/conll2000/chunking/test.txt.gz
NOTE: No known validation data set is available.

Data sets and word embedding vectors are downloaded when main.py is run (if they are not present locally).
The glove.42B word embedding vectors are used by default (http://nlp.stanford.edu/data/glove.42B.300d.zip).

Using the --birnn option with other default option values yields a model with 
testing accuracy of ~97%, averaged between the accuracy of POS tagging and chunking tasks.
This number is arguably inflated because the model uses batches of sentences which are heavily padded, 
and the associated pad tokens are readily tagged and mostly irrelevant to the underlying task.

For training and testing, a progress bar is provided to monitor running accuracy and completion of the epoch.

The model is saved after each epoch, upon forceful exit (keyboard interrupt) during training, 
or if time spent training exceeds the --timeout option value.

The model is loaded if present (unless --fresh option is provided, indicating a new model should be used).

As an additional point of testing, the --wiki option permits tagging of sentences from wikipedia pages.
Currently by default this only attempts to tag sentences of the "Python (programming language)" wikipedia page.
Performance is hampered by relatively disjoint vocabularies between this page and the Wall Street Journal corpus.
Despite this, the model appears to provide mostly accurate tagging. 


TODO:

  Testing on GPUs (no CUDA available on system on which chunktagger was developed).

