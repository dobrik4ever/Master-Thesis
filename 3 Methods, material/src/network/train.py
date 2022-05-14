import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import ResNet
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv',sep=';')
data_train, data_valid = train_test_split(df, train_size = 0.75, test_size = 0.25)
dataset_train = ChallengeDataset(data_train, mode='train')
dataset_valid = ChallengeDataset(data_valid, mode='val')

bs = 32
dataset_train = t.utils.data.DataLoader(dataset_train,
                                        batch_size = bs,
                                        shuffle = True)

dataset_valid = t.utils.data.DataLoader(dataset_valid,
                                        batch_size = bs,
                                        shuffle = True)

model = ResNet()
loss = t.nn.BCELoss()
learning_rate = 1e-3
optimizer = t.optim.SGD(model.parameters(), lr = learning_rate)
trainer = Trainer(model, loss, optimizer, dataset_train, dataset_valid, True, 5)
res = trainer.fit()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')