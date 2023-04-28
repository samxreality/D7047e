from sklearn.model_selection import train_test_split
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from model import *
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

chunk_len = 100
print()
dataset = []
k = 0
for char in text:
    ind = [dictionary.index(char)]
    dataset.append(ind)

dataset = torch.tensor(dataset).to(device)

########## Hyperparameters ###########
hidden_size = 200   # size of hidden state
seq_len = 100       # length of LSTM sequence
num_layers = 3      # num of layers in LSTM layer stack
learning_rate = 0.001          # learning rate
weight_decay = 1e-5            #weight decay
epochs = 20        # max number of epochs
drop_prob = 0.5
vocab_size = len(dictionary)  #length pf dictionary

#initialize network
wordRNN = WordRNN(vocab_size, vocab_size, hidden_size, num_layers, drop_prob)
wordRNN = wordRNN.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wordRNN.parameters(), lr=learning_rate, weight_decay=weight_decay)

#####TRAINING#####

total_train_loss = []
loss = 0
for i in range(epochs):
    train_loss = 0
    n = 0
    data_ptr = np.random.randint(100)
    while True:
        hidden_state = wordRNN.initHidden(1)
        input_sequence = dataset[data_ptr:data_ptr + seq_len]
        target_sequence = dataset[data_ptr + 1:data_ptr + seq_len + 1]

        out, hidden_state = wordRNN(input_sequence, hidden_state)

        loss = criterion(torch.squeeze(out), torch.squeeze(target_sequence))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        data_ptr += seq_len
        n += 1

        print(f"Epoch [{i + 1}]/[{epochs}] loss: {loss.item()}", end="\r", flush=True)

        if data_ptr + seq_len + 1 > data_size:
            break
    total_train_loss.append(train_loss / n)

    print(f"Epoch loss: {train_loss / n}")
    print("-----------------------------------------------------------------")

    hidden_state = wordRNN.initHidden(1)
    rand_index = np.random.randint(data_size - 1)
    input_sequence = dataset[rand_index: rand_index + 1]
    for i in range(100):
        out, hidden_state = wordRNN(input_sequence, hidden_state)

        # construct categorical distribution and sample a character
        out = F.softmax(torch.squeeze(out), dim=0)
        dist = Categorical(out)
        index = dist.sample()

        print(dictionary[index.item()], end="")

        input_sequence[0][0] = index.item()
    print("\n-----------------------------------------------------------------\n")

torch.save(wordRNN, "bestmodel")
