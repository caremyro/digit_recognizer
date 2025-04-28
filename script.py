import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#schema descriptif de mon reseau de neurones : 
#couche 0 -> entrees (image de 28x28 pixels, soit 784 pixels)
#couche 1 (hidden layer) -> activations avec ReLU
#couche 2 (hidden layer) -> activations avec Softmax
#couche 3 -> sorties -> resultat final 

data = pd.read_csv('train.csv') #on recupere les donnees du fichier
m, n = data.shape #initialisation des variables relatives aux lignes et colonnes de la matrice
np.random.shuffle(data) #melange des donnees 

#separation dev et train afin de detecter du overfitting : 

data_dev = data[0:1000].T #transpose des 999 premieres lignes pourquoi ? colonne = exemple | ligne = caracteristique (pixel, valeur, ...)
Y_dev = data_dev[0] #cela correspond aux reponses (les classes a predire)
X_dev = data_dev[1:c] #cela correspond aux donnees pour entrainer ou tester le modele
# X_dev = X_dev / 255. #normalisation des donnees (entre 0 et 1)  

data_train = data[1000:l].T
Y_train = data_train[0]
X_train = data_train[1:c]
# X_train = X_train / 255.

#definition de la methode ReLU qui permet de rendre la fonction non lineaire et donc rajoute de la complexite a notre reseau de neurone

def ReLU(Z):
    return max(Z, 0) 
    # if x > 0 : 
    #     return x
    # else : 
    #     return 0

#definition de la fonction softmax (fonction exponentielle normalisee), convertit un vecteur de K nombres reels en une distribution de probabilites sur K choix

def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))

#definition de la fonction one_hot, represente les classes dans un matrice binaire, l'indice remplace par un 1 correspond a la classe qui devrait etre predite

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))   # Crée une matrice de zéros
    one_hot_Y[np.arange(Y.size), Y] = 1           # Remplace les indices correspondants à Y par 1
    one_hot_Y = one_hot_Y.T                       # Transpose la matrice
    return one_hot_Y

def ReLU_deriv(Z):
    return Z > 0

#definition de la fonction de propagation avant qui permet selon une entree de sortir un resultat ou une prediction.
#donc pour calculer les resultats Z et les activations A, il nous faut les poids W, biais B et l'entree X

def forward_propagation(W1, W2, B1, B2, X): 
    Z1 = W1.dot(X) + B1 # produit matriciel entre l'entree et le poids avec l'ajout du biais 
    A1 = ReLU(Z1) # calcul des activations de la couche 1 (premiere couche cachee)
    Z2 = W2.dot(A1) + B2 #produit matriciel des poids avec les resultats (actives) et non les resultats brutes et bien sur l'ajout du biais
    A2 = softmax(Z2) # calcul des activations avec softmax
    return Z1, A1, Z2, A2

#definition de la fonction de propagation arriere, qui permet l'apprentissage automatique, calcul les erreurs de la derniere couche a la premiere couche et met a jour les poids et les biais afin d'ameliorer ses predictions 

def backward_propagation(A1, Z1, A2, W2, Z2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # calcul de l'erreur de la couche de sortie, on fait la difference entre les predictions et les etiquettes 
    dW2 = 1/m * dZ2.dot(A1.T) # calcul du gradient des poids de la couche 2
    dB2 = 1/m * sum(dZ2) # calcul du gradient des biais de la couche 2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) #calcul de l'erreur de la couche 1 en prenant en compte la propagation de l'erreur de la couche 2 a la couche 1 et de l'activation des neurones dans la couche 1 
    dW1 = 1/m * dZ1.dot(X.T) # calcul du gradient des poids de la couche 1
    dB1 = 1/m * sum(dZ1) # calcul du gradient des biais de la couche 1
    return dZ2, dW2, dB2, dZ1, dW1, dB1

#ici on va mettre a jour simplement les parametres en prenant en compte les erreur, alpha est le taux d'apprentissage 
# si alpha est trop grand, l'apprentissage peut etre instable par la mise a jour agressive des poids et biais 
# si alpha est trop petit, l'apprentissage sera trop lent et convergera apres de nombreux iterations

def update_params(W1, B1, dW1, dB1, W2, B2, dW2, dB2, alpha): 
    W1 = W1 - alpha * dW1
    B1 = B1 - alpha * dB1
    W2 = W2 - alpha * dW2
    B2 = B2 - alpha * dB2
    return W1, B1, W2, B2

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

