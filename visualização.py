# -*- coding: utf-8 -*-
from load_dataset import load_data
import matplotlib.pyplot as plt
import os

path = os.getcwd() #Diretório do programa

#Carrega as 60000 instancias de treino
X_train, y_train = load_data(path, tipo='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

#Carrega as 10000 instancias de teste
X_test, y_test = load_data(path, tipo='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#Reestruturando o vetor de 28 pixels em imagens para visualização
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    #pegar algarismo 0 até 9, muda o vetor para uma matriz 28x28
    img = X_train[y_train ==i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
ax[0].set_xticks([])
ax[0].set_yticks([])

#Mostra o plot dos 10 algarismo
print("\n\n\n\nNúmeros de 0 a 9:")
plt.tight_layout()
plt.show()

#Exemplos de algarismo '3'
fig, ax = fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):
    #Coleta 25 instancias
    img = X_train[y_train == 3][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    
ax[0].set_xticks([])
ax[0].set_yticks([])

print("\n\n\n\n25 instâncias de '3':")
plt.tight_layout()
plt.show()

