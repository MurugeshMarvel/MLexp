import numpy 
import os
from collections import Counter
train_dir = './train-mails'
test_dir = './test-mails'

def create_dict(root_dir):
	all_words = []
	emails = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
	for email in emails:
		with open(email) as m:
			for line in m:
				words = line.split()
				all_words += words
	dictionary = Counter(all_words)
	list_to_remove = list(dictionary.keys())
	for item in list_to_remove:
		if item.isalpha() == False:
			del dictionary[item]
		elif len(item) == 1:
			del dictionary[item]
	dictionary = dictionary.most_common()
	print dictionary
create_dict(train_dir)
