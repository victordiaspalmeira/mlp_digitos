from mlp import NeuralNetMLP
from load_dataset import load_data
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
path = os.getcwd()
X_train, y_train = load_data(path, tipo='train')
X_test, y_test = load_data(path, tipo='t10k')

""" Abre o arquivo do objeto rede neural """
with open('neural_network.pkl', 'rb') as input_:
    neural_network = pickle.load(input_)

""" Verificando acurácia da rede para o conjunto de treinamento """
y_train_pred = neural_network.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print('Acurácia para o conjunto de treinamento: %.2f%%' % (acc * 100))

""" Verificando acurácia da rede para o conjunto de testes """
y_test_pred = neural_network.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print('Acurácia para o conjunto de teste: %.2f%%' % (acc * 100))

""" Exemplos de classificação correta"""
print("\n\n\nClassificações corretas: [0]")
cl_img = X_test[y_test == y_test_pred][:25]
correct_lab = y_test[y_test == y_test_pred][:25]
cl_lab= y_test_pred[y_test == y_test_pred][:25]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = cl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], cl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações corretas: [1]")
cl_img = X_test[y_test == y_test_pred][25:50]
correct_lab = y_test[y_test == y_test_pred][25:50]
cl_lab= y_test_pred[y_test == y_test_pred][25:50]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = cl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], cl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações corretas: [2]")
cl_img = X_test[y_test == y_test_pred][50:75]
correct_lab = y_test[y_test == y_test_pred][50:75]
cl_lab= y_test_pred[y_test == y_test_pred][50:75]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = cl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], cl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações corretas: [3]")
cl_img = X_test[y_test == y_test_pred][75:100]
correct_lab = y_test[y_test == y_test_pred][75:100]
cl_lab= y_test_pred[y_test == y_test_pred][75:100]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = cl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], cl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações corretas: [4]")
cl_img = X_test[y_test == y_test_pred][100:125]
correct_lab = y_test[y_test == y_test_pred][100:125]
cl_lab= y_test_pred[y_test == y_test_pred][100:125]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = cl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], cl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
print("\n\n\nClassificações corretas: [5]")
cl_img = X_test[y_test == y_test_pred][125:150]
correct_lab = y_test[y_test == y_test_pred][125:150]
cl_lab= y_test_pred[y_test == y_test_pred][125:150]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = cl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], cl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
""" """

""" Exemplos de classificação incorreta"""
print("----------------------------------")
print("\n\n\nClassificações incorretas: [0]")
miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab= y_test_pred[y_test != y_test_pred][:25]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações incorretas: [1]")
miscl_img = X_test[y_test != y_test_pred][25:50]
correct_lab = y_test[y_test != y_test_pred][25:50]
miscl_lab= y_test_pred[y_test != y_test_pred][25:50]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações incorretas: [2]")
miscl_img = X_test[y_test != y_test_pred][50:75]
correct_lab = y_test[y_test != y_test_pred][50:75]
miscl_lab= y_test_pred[y_test != y_test_pred][50:75]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações incorretas: [3]")
miscl_img = X_test[y_test != y_test_pred][75:100]
correct_lab = y_test[y_test != y_test_pred][75:100]
miscl_lab= y_test_pred[y_test != y_test_pred][75:100]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações incorretas: [4]")
miscl_img = X_test[y_test != y_test_pred][100:125]
correct_lab = y_test[y_test != y_test_pred][100:125]
miscl_lab= y_test_pred[y_test != y_test_pred][100:125]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

print("\n\n\nClassificações incorretas: [5]")
miscl_img = X_test[y_test != y_test_pred][125:150]
correct_lab = y_test[y_test != y_test_pred][125:150]
miscl_lab= y_test_pred[y_test != y_test_pred][125:150]
fig, ax = plt.subplots(nrows=5,ncols=5,sharex=True,sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img,cmap='Greys',interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d'% (i+1, correct_lab[i], miscl_lab[i]))
    
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

"""
Conclusões:
    1. Conseguimos uma rede com boa capacidade de identificação dos caracteres.
    2. A identificação para o conjunto de treinamento foi ligeiramente melhor,
    indicando que existe overfitting (sobreajuste), mas este tem valor pequeno.
    3. Apesar de a rede neural falhar em identificar caracteres que (para humanos)
    são óbvios, tem caracteres que nem a gente consegue identificar direito.
"""