
from sklearn.model_selection import train_test_split
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model import *

wordRNN = torch.load("bestmodel")
wordRNN = wordRNN.to(device)

special_characters = [' ', ':', '\n', ',', '.', '?', "'", ';', '!', '-', '&', '3']

f = open("tinyshakespeare.txt", "r")
text = f.read()
data_size = len(text)

dictionary = []
dataset = []

j = 0
for i in range(data_size):
    char = text[i]
    if char in special_characters:
        word = text[j:i]
        if i == j:
            word = text[j:i + 1]

        if char not in dictionary:
            dictionary.append(char)

        if word not in dictionary:
            dictionary.append(word)

        if i != j:
            charInd = dictionary.index(char)
            wordInd = dictionary.index(word)
            dataset.append([wordInd])
            dataset.append([charInd])
        else:
            wordInd = dictionary.index(word)
            dataset.append([wordInd])

        j = i + 1
    continue
data_size = len(dataset)
dataset = torch.tensor(dataset)

device = torch.device("cuda")


#Preprocessing
f = open("tinyshakespeare.txt", "r")
text = f.read()

data_size = len(text)

dictionary = []
for i in text:
    if i not in dictionary:
        dictionary.append(i)
#print(dictionary)


def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        try:
            tensor[c] = dictionary.index(string[c])
        except:
            continue
    return tensor

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.initHidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        inp = torch.unsqueeze(prime_input[:, p], dim=1).to(device)
        out, hidden = decoder(inp, hidden)

    inp = torch.unsqueeze(prime_input[:, -1], dim=1).to(device)

    # hidden = (Variable(hidden[0].detach().to(device)), Variable(hidden[1].detach().to(device)))

    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = dictionary[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0)).to(device)
        if cuda:
            inp = inp.cuda()

    return predicted


primers = ["The", ".", "which is,", "blah blah blah"]

for word in primers:
    print("primer: "+word+"\n")
    print(generate(wordRNN, prime_str=word, cuda=True))
    print("\n---------------------------------------------\n")
