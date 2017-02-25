'''
SingleTon Access for the trained Doc2Vec Model
'''
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', filename="doc2vecModelInfoLog.log", level=logging.INFO)
from gensim.models.doc2vec import *
import MeCab
from scipy.spatial.distance import cosine

#class Singleton(object):
#    def __new__(cls,*args,**kwargs):
#        if not hasattr(cls,'_inst'):
#            cls._inst=super(Singleton,cls).__new__(cls,*args,**kwargs)
#        return cls._inst
#class DocModel(Singleton):
#    def __init__(self,filename):
#        self.model=Doc2Vec.load(filename)
languageModel=Doc2Vec.load("model/bccwj_vec")
def getVector(s):
    mec=MeCab.Tagger("-Ochasen")
    inp=[]
    if not s:
        return None
    node=mec.parseToNode(s)
    while node:
        #print node.surface+"\n"
        inp.append(node.surface)
        node=node.next
    return languageModel.infer_vector(inp)
def getCos(a,b):
    return cosine(a,b)
