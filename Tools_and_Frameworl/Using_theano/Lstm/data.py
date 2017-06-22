dataset_path = '/home/murugesan/DEV/My Repos/Artintell/Deep_Learning/Using_theano/Lstm/Datasets/aclImdb/'

import numpy as np
import cPickle as pkl
from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

tokenizer_cmd = ['sudo','./tokenizer.perl','-l','en','-q','-']

def tokenize(sentences):
	print 'Tokenising../././'
	text = "\n".join(sentences)
	tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
	tok_text, _ =tokenizer.communicate(text)
	toks = tok_text.split('\n')[:-1]
	print 'Done'

def build_dict(path):
	sentences = []
	currdir = os.getcwd()
	os.chdir('%s/pos/' % path)
	for f in glob.glob("*.txt"):
		with open(f,'r') as file:
			sentences.append(file.readline().strip())
	os.chdir("%s/neg/" %path)
	for f in glob.glob("*.txt"):
		with open(f,'r') as file:
			sentences.append(file.readline().strip())
	os.chdir(currdir)
	sentences = tokenize(sentences)
	print sentences
	print "Building Dictionary../././"
	wordcount = dict()
	for s in sentences:
		words = s.strip().lower().split()
		for w in words:
			if w not in wordcount:
				wordcount[w] = 1
			else:
				wordcount[w]+=1
	counts = wordcount.values()
	keys = wordcount.keys()
	sorted_idx = np.argsort(counts)[::-1]
	worddict = dict()
	for idx, s in enumerate(sorted_idx):
		worddict[keys[s]] = idx+2 
	print np.sum(counts), 'totals words',len(keys),'unique words'
	return worddict

def grab_data(path, dictionary):
	sentences = []
	currdir = os.getcwd()
	os.chdir(path)
	for f in glob.glob("*.txt"):
		with open(f, 'r') as file:
			sentences.append(file.readline().split())
	os.chdir(currdir)
	sentences = tokenize(sentences)

	seqs = [None] * len(sentences)
	for idx, s in enumerate(sentences):
		words = ss.strip().lower().split()
		seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]
	return seqs
def main():
	path = dataset_path
	dictionary = build_dict(os.path.join(path, 'train'))
	train_x_pos = grab_data(path+'train/pos', dictionary)
	train_x_neg = grab_data(path+'train/neg', dictionary)
	train_x = train_x_pos + train_x_neg
	train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

	test_x_pos = grab_data(path+'test/pos', dictionary)
	test_x_neg = grab_data(path+'test/neg', dictionary)
	test_x = test_x_pos + test_x_neg
	test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

	f = open('imdb.pkl', 'wb')
	pkl.dump((train_x, train_y), f, -1)
	pkl.dump((test_x, test_y),f,-1)
	f.close()

if __name__ == '__main__':
	main()