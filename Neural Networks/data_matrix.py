
import nltk
import itertools
import csv
Voc_size = 8000
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
