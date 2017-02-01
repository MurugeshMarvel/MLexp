Voc_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


'''reading data'''
print "Reading CSV file...."
with open('data/comment-2015-08.csv','rb') as f:
	reader = csv.reader(f, skipinitialspace = True)
	reader.next()
	sentences = itertools.