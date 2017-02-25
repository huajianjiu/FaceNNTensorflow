import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models.doc2vec import *

if __name__=="__main__":
    documents=TaggedLineDocument("BCCWJptwakati-linux.txt")
    model = Doc2Vec(documents, size=800, window=8, min_count=5, workers=4)
    model.save("bccwj_vec")
