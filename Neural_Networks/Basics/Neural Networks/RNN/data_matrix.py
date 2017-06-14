
import nltk
import itertools
import numpy as np
import csv


def data():
	voc_size = 8000
	unknown_token = "UNKNOWN_TOKEN"
	sentence_start_token = "SENTENCE_START"
	sentence_end_token = "SENTENCE_END"


	'''reading data'''
	print "Reading CSV file...."
	with open('comments-2015-08.csv','rb') as f:
		reader = csv.reader(f, skipinitialspace = True)
		reader.next()
		sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower())for x in reader])
		sentences = ["%s%s%s" % (sentence_start_token,x,sentence_end_token)for x in sentences]
	print "Parsed %d sentences" % (len(sentences))

	#tokenise the sentence into words

	tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

	#count the word frequencies

	word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
	print "Found %d unique words tokens: " % len(word_freq.items())

	#Get the most common words and build index_to_word and word_to_index vectors


	vocab = word_freq.most_common(voc_size-1)
	index_to_word = [x[0] for x in vocab]
	index_to_word.append(unknown_token)
	word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

	print "Using Vocabulary size %d:" % voc_size
	print vocab[-1][0]
	print "The least frequent word in our vocabulary is ",vocab[-1][0],"and appeared %d times ",vocab[-1][1]


	for i,sent in enumerate(tokenized_sentences):
		tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

	print "\n Example Sentence :%s" %sentences[0] 
	print "\n Example Sentence after Pre processing : %s" %tokenized_sentences[0]

	#Creating Training Data
	print "Creating Training data"
	x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
	y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

	file = open("training data","a")
	for i in range(0,len(x_train)):
		val = '\n'+ str(x_train[i]) +'----'+ str(y_train[i])
		file.write(val)

	file.close()
	print "Training Data saved" 
	return voc_size,x_train