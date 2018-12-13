# -*- coding: utf-8 -*-
from mlp import NeuralNetMLP
from load_dataset import load_data
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

#Carrega o dataset de treino e teste
path = os.getcwd()
#Carrega as 60000 instancias de treino
X_train, y_train = load_data(path, tipo='train')
print('Rows: %d, columns: %d' % (X_train.shape[0], X_train.shape[1]))

#Carrega as 10000 instancias de teste
X_test, y_test = load_data(path, tipo='t10k')
print('Rows: %d, columns: %d' % (X_test.shape[0], X_test.shape[1]))

#Cria uma instância de rede neural com os seguintes parâmetros
neural_network = NeuralNetMLP(n_output=10,
                              n_features=X_train.shape[1],
                              n_hidden=50,
                              l2=0.1,
                              l1=0.0,
                              epochs=1000,
                              eta=0.001,
                              alpha=0.001,
                              decrease_const=0.00001,
                              shuffle=True,
                              minibatches=50,
                              random_state=1)


#Treina a rede neural a partir das 60000 instâncias

""" Tirar do comentário para executar o fitting """
#neural_network.fit(X_train, y_train, print_progress=True)

""" Tirar do comentário para salvar objeto rede neural em disco """
#with open('neural_network.pkl', 'wb') as output:
#    pickle.dump(neural_network, output, pickle.HIGHEST_PROTOCOL)

""" Tirar do comentário para abrir objeto rede neural em disco """
with open('neural_network.pkl', 'rb') as input_:
    neural_network = pickle.load(input_)
    
#Visualização da convergência do algoritmo, levando em conta os 50 minibatches
batches = np.array_split(range(len(neural_network.cost_)), 1000)
cost_ary = np.array(neural_network.cost_)
print(cost_ary)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]


plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Custo')
plt.xlabel('Épocas')
plt.tight_layout()
plt.show()


