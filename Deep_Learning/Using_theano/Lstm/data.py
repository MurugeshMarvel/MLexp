dataset_path = 'Datasets/aclImdb'

import numpy as np
import cPickle as pkl
from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

tokenizer_cmd = ['./tokenizer.perl','-l','en','-q','-']

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
			