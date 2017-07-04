import csv
import itertools
import operator
import numpy as np
import nltk
import sys
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt













if __name__ == '__main__': # should make it a conditional statement if book corpus is not present
	#nltk.download("book")
	vocabulary_size = 8000
	unknown_token = "UNKNOWN_TOKEN"
	sentence_start_token = "SENTENCE_START"
	sentence_end_token = "SENTENCE_END"
	print "Reading the CSV file"
	with open('data/reddit-comments.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace = True)
		reader.next()
		sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower())for x in reader])
		sentences = ["%s %s %s"%(sentence_start_token,x,sentence_end_token) for x in sentences]
	print "Parsed %d sentences" %(len(sentences))
	tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	print "Found %d unique words tokens. "%len(word_freq.items())
	vocab = word_freq.most_common(vocabulary_size -1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
	print "Using Vocabulary  size %d" %vocabulary_size
	print "The least Frequent word in our vocabulary is '%s' and appeared %d times."%(vocab[-1][0], vocab[-1][1])
	for i, sent in enumerate(tokenized_sentences):
		tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

	print "\n Example Sentences: '%s'"% sentences[0]
	print "\n Example Sentence after Pre-processing: '%s'" %tokenized_sentences[0]

	x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent])