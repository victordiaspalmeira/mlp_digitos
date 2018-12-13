# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:12:01 2018

@author: ASUS
"""

import os, struct
import numpy as np

"""
Cada imagem do dataset é uma matriz de 28x28 pixels.
Cada pixel pode ser representado por um grau de intensidade da escala de cinza.
""



Conjunto de treino:     60.000 instancias
Conjunto de teste:      10.000 instancias
"""

def load_data(path, tipo='train'):
    """Carrega os dados da raiz"""
    
    #Carrega o caminho dos labels
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % tipo)
    
    #Carrega o caminho das imagens
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % tipo) 

    with open(labels_path, 'rb') as lbpath:
        #Abre o arquido das labels
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype = np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        #Abre o arquivo de imagem
        #Cada imagem de 28x28 pixels pode ser organizada em um vetor unidimensional de 784 indices.
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype = np.uint8).reshape(len(labels), 784)
        
    #Obs: Magic, ou magic number é a descrição do protocolo de arquivo
    
    #Retorna dois arrays:
        #1. Images (nxm) n - número de instancias, m - numero de features
        #2. Labels das classes - Algarismos de 0 a 9
    return images, labels