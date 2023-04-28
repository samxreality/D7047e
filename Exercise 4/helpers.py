# https://github.com/spro/char-rnn.pytorch

import unidecode
import string
import random
import time
import math
import torch
import nltk
nltk.download('punkt')

# Reading and un-unicode-encoding data

all_characters = string.printable
n_characters = len(all_characters)

def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)

def read_file_word(filename):
    with open(filename, 'r') as f:
        text = f.read()
    words = nltk.word_tokenize(text)
    return words, len(words)

# Turning a string into a tensor

def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = all_characters.index(string[c])
        except:
            continue
    return tensor

def word_tensor(words, all_words):
    tensor = torch.zeros(len(words)).long()
    for i, word in enumerate(words):
        try:
            tensor[i] = all_words.index(word)
        except:
            continue
    return tensor
# Readable time elapsed

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

