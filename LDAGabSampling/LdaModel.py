from ipdb import set_trace
import numpy as np
import random
import logging
import logging.config
import ConfigParser
import codecs
import os
from collections import OrderedDict


# Code initialization
path = os.getcwd()
logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

# load setting file
conf = ConfigParser.ConfigParser()
conf.read("setting.conf") 

# file path 
# trainfile = os.path.join(path,os.path.normpath(conf.get("filepath", "trainfile")))
# wordidmapfile = os.path.join(path,os.path.normpath(conf.get("filepath","wordidmapfile")))
# thetafile = os.path.join(path,os.path.normpath(conf.get("filepath","thetafile")))
# phifile = os.path.join(path,os.path.normpath(conf.get("filepath","phifile")))
# paramfile = os.path.join(path,os.path.normpath(conf.get("filepath","paramfile")))
# topNfile = os.path.join(path,os.path.normpath(conf.get("filepath","topNfile")))
# tassginfile = os.path.join(path,os.path.normpath(conf.get("filepath","tassginfile")))
trainfile = path+os.path.normpath(conf.get("filepath", "trainfile"))
wordidmapfile = path+os.path.normpath(conf.get("filepath","wordidmapfile"))
thetafile = path+os.path.normpath(conf.get("filepath","thetafile"))
phifile = path+os.path.normpath(conf.get("filepath","phifile"))
paramfile = path+os.path.normpath(conf.get("filepath","paramfile"))
topNfile = path+os.path.normpath(conf.get("filepath","topNfile"))
tassginfile = path+os.path.normpath(conf.get("filepath","tassginfile"))


# set_trace()

# parameter
K = int(conf.get("model_args","K"))
alpha = float(conf.get("model_args","alpha"))
beta = float(conf.get("model_args","beta"))
iter_times = int(conf.get("model_args","iter_times"))
top_words_num = int(conf.get("model_args","top_words_num"))

class Document(object):

	def __init__(self):
		self.words = []
		self.length = 0

class DataPreProcessing(object):

	def __init__(self):
		self.docs_count = 0
		self.words_count = 0
		self.docs = []
		self.word2id = OrderedDict()

	def cachewordidmap(self):
		with codecs.open(wordidmapfile,'w','utf-8') as f:
			for word,id in self.word2id.items():
				f.write(word+"\t"+str(id)+"\n")

class LDAModel(object):

	def __init__(self,dpre):
		# get preprocess parameter
		self.dpre = dpre 

		#top_words_num is each topic's words num
		self.K = K
		self.beta = beta
		self.alpha = alpha
		self.iter_times = iter_times
		self.top_words_num = top_words_num

		# trainfile is the file after segment
		# wordidmapfile is the word  : id
		# thetafile is doc -> topic distribution
		# phifile is word -> topic distribution
		# topNfile is each topic topN words
		# tassginfile is the final result
		# paramfile save the parameter
		self.wordidmapfile = wordidmapfile
		self.trainfile = trainfile
		self.thetafile = thetafile
		self.phifile = phifile
		self.topNfile = topNfile
		self.tassginfile = tassginfile
		self.paramfile = paramfile

		# nw word at topic disctribution
		# nwsum each topic words sum
		# nd each doc each topic words sum
		# ndsum each doc words sum
		# Z M*doc.size() earch words at doc topic distribution
		self.p = np.zeros(self.K)        
		self.nw = np.zeros((self.dpre.words_count,self.K),dtype="int")       
		self.nwsum = np.zeros(self.K,dtype="int")    
		self.nd = np.zeros((self.dpre.docs_count,self.K),dtype="int")       
		self.ndsum = np.zeros(dpre.docs_count,dtype="int")    
		self.Z = np.array([ [0 for y in xrange(dpre.docs[x].length)] for x in xrange(dpre.docs_count)])  

		for x in xrange(len(self.Z)):
			self.ndsum[x] = self.dpre.docs[x].length
			for y in xrange(self.dpre.docs[x].length):
				topic = random.randint(0,self.K-1)
				self.Z[x][y] = topic
				self.nw[self.dpre.docs[x].words[y]][topic] += 1
				self.nd[x][topic] += 1
				self.nwsum[topic] += 1

		self.theta = np.array([ [0.0 for y in xrange(self.K)] for x in xrange(self.dpre.docs_count) ])
		self.phi = np.array([ [ 0.0 for y in xrange(self.dpre.words_count) ] for x in xrange(self.K)]) 

	def sampling(self,i,j):

		topic = self.Z[i][j]
		word = self.dpre.docs[i].words[j]
		self.nw[word][topic] -= 1
		self.nd[i][topic] -= 1
		self.nwsum[topic] -= 1
		self.ndsum[i] -= 1

		Vbeta = self.dpre.words_count * self.beta
		Kalpha = self.K * self.alpha
		self.p = (self.nw[word] + self.beta)/(self.nwsum + Vbeta) * (self.nd[i] + self.alpha) / (self.ndsum[i] + Kalpha)

		for k in xrange(1,self.K):
			self.p[k] += self.p[k-1]

		u = random.uniform(0,self.p[self.K-1])
		for topic in xrange(self.K):
			if self.p[topic] > u:
				break

		self.nw[word][topic] += 1
		self.nwsum[topic] +=1
		self.nd[i][topic] +=1
		self.ndsum[i] +=1

		return topic

	def est(self):

		for x in xrange(self.iter_times):
			for i in xrange(self.dpre.docs_count):
				for j in xrange(self.dpre.docs[i].length):
					topic = self.sampling(i,j)
					self.Z[i][j] = topic

		logger.info("finish iteration")
		logger.debug("cal doc-topic distribution")
		self._theta()
		logger.debug("cal words-topic distribution")
		self._phi()
		logger.debug("save model")
		self.save()

	def _theta(self):
		for i in xrange(self.dpre.docs_count):
			self.theta[i] = (self.nd[i]+self.alpha)/(self.ndsum[i]+self.K * self.alpha)

	def _phi(self):
		for i in xrange(self.K):
			self.phi[i] = (self.nw.T[i] + self.beta)/(self.nwsum[i]+self.dpre.words_count * self.beta)

	def save(self):
		with codecs.open(self.thetafile,'w') as f:
			for x in xrange(self.dpre.docs_count):
				for y in xrange(self.K):
					f.write(str(self.theta[x][y])+'\t')
				f.write('\n')
		logger.info('save the thetafile')

		with codecs.open(self.phifile,'w') as f:
			for x in xrange(self.K):
				for y in xrange(self.dpre.words_count):
					f.write(str(self.phi[x][y])+'\t')
				f.write('\n')
		logger.info('save the phifile')

		with codecs.open(self.paramfile,'w','utf-8') as f:
			f.write('K=' + str(self.K) + '\n')
			f.write('alpha=' + str(self.alpha) + '\n')
			f.write('beta=' + str(self.beta) + '\n')
			f.write('iter_times=' + str(self.iter_times) + '\n')
			f.write('top_words_num=' + str(self.top_words_num) + '\n')
		logger.info('save the paramfile')

		with codecs.open(self.topNfile,'w','utf-8') as f:
			self.top_words_num = min(self.top_words_num,self.dpre.words_count)
			for x in xrange(self.K):
				f.write(str(x)+ ':\n')
				twords = []
				twords = [(n,self.phi[x][n]) for n in xrange(self.dpre.words_count)]
				twords.sort(key = lambda i:i[1], reverse= True)
				for y in xrange(self.top_words_num):
					word = OrderedDict({value:key for key, value in self.dpre.word2id.items()})[twords[y][0]]
					f.write('\t'*2+ word +'\t' + str(twords[y][1])+ '\n')
		logger.info('save the topNfile')

		with codecs.open(self.tassginfile,'w') as f:
			for x in xrange(self.dpre.docs_count):
				for y in xrange(self.dpre.docs[x].length):
					f.write(str(self.dpre.docs[x].words[y])+':'+str(self.Z[x][y])+ '\t')
				f.write('\n')
		logger.info('save the tassginfile')

		logger.info('finish train model')

def preprocessing():
	logger.info('load data')
	with codecs.open(trainfile,'r','utf-8') as f:
		docs = f.readlines()
	logger.info('generate vocab and dict')
	dpre = DataPreProcessing()
	items_idx = 0
	for line in docs:
		if line != "":
			tmp = line.strip().split()
			doc = Document()
			for item in tmp:
				if dpre.word2id.has_key(item):
					doc.words.append(dpre.word2id[item])
				else:
					dpre.word2id[item] = items_idx
					doc.words.append(items_idx)
					items_idx += 1
			doc.length = len(tmp)
			dpre.docs.append(doc)
		else:
			pass

	dpre.docs_count = len(dpre.docs)
	dpre.words_count = len(dpre.word2id)
	logger.info('total have %s docs' % dpre.docs_count)
	dpre.cachewordidmap()
	logger.info('save the wordidmapfile')

	return dpre 

def run():
	dpre = preprocessing()
	lda = LDAModel(dpre)
	lda.est()

if __name__ == '__main__':
	run()






