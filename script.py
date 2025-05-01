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
data = data.sample(frac=1).reset_index(drop=True) #melange des donnees 

#separation dev et train afin de detecter du overfitting : 

# On prend les 1000 premières lignes pour le dev set
data_dev = data.iloc[:1000]
Y_dev = data_dev.iloc[:, 0]  # Première colonne = labels
X_dev = data_dev.iloc[:, 1:]  # Reste des colonnes = pixels
X_dev = X_dev.T  # Transpose pour avoir les exemples en colonnes
X_dev = X_dev / 255.  # Normalisation

# Le reste pour le train set
data_train = data.iloc[1000:]
Y_train = data_train.iloc[:, 0]  # Première colonne = labels
X_train = data_train.iloc[:, 1:]  # Reste des colonnes = pixels
X_train = X_train.T  # Transpose pour avoir les exemples en colonnes
X_train = X_train / 255.  # Normalisation

# Vérification des dimensions
print("Shape de X_train:", X_train.shape)
print("Shape de Y_train:", Y_train.shape)
print("Nombre d'exemples d'entraînement:", X_train.shape[1])
print("Nombre d'exemples de test:", X_dev.shape[1])

#definition de la methode ReLU qui permet de rendre la fonction non lineaire et donc rajoute de la complexite a notre reseau de neurone

def ReLU(Z):
    return np.maximum(Z, 0) 

#definition de la fonction softmax (fonction exponentielle normalisee), convertit un vecteur de K nombres reels en une distribution de probabilites sur K choix

def softmax(Z):
    exp = np.exp(Z - np.max(Z, axis=0))
    return exp / np.sum(exp, axis=0)

#definition de la fonction one_hot, represente les classes dans un matrice binaire, l'indice remplace par un 1 correspond a la classe qui devrait etre predite

def one_hot(Y):
    Y = np.array(Y, dtype=np.int32)  # s'assurer que c'est un ndarray d'entiers
    if Y.size == 0:
        raise ValueError("Y ne peut pas être vide")
    one_hot_Y = np.zeros((Y.size, 10), dtype=np.float32)  # 10 classes pour MNIST
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

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
    m = X.shape[1]  # nombre d'exemples
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y # calcul de l'erreur de la couche de sortie, on fait la difference entre les predictions et les etiquettes 
    dW2 = 1/m * dZ2.dot(A1.T) # calcul du gradient des poids de la couche 2
    dB2 = 1/m * np.sum(dZ2, axis=1, keepdims=True) # calcul du gradient des biais de la couche 2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) #calcul de l'erreur de la couche 1 en prenant en compte la propagation de l'erreur de la couche 2 a la couche 1 et de l'activation des neurones dans la couche 1 
    dW1 = 1/m * dZ1.dot(X.T) # calcul du gradient des poids de la couche 1
    dB1 = 1/m * np.sum(dZ1, axis=1, keepdims=True) # calcul du gradient des biais de la couche 1
    return dW2, dB2, dW1, dB1

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
    W1 = np.random.rand(10, 784) * 0.01
    B1 = np.zeros((10, 1))
    W2 = np.random.rand(10, 10) * 0.01
    B2 = np.zeros((10, 1))
    return W1, B1, W2, B2

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_predictions(A2):
    return np.argmax(A2, 0)

def gradient_descent(X, Y, iterations, alpha): #entrainement du reseau de neurones
    W1, B1, W2, B2 = init_params() # On initialise simplement les parametres dont on a besoin pour la suite
    for i in range(iterations): 
        Z1, A1, Z2, A2 = forward_propagation(W1, W2, B1, B2, X) #on fait en premier une propagation en avant
        dW2, dB2, dW1, dB1 = backward_propagation(A1, Z1, A2, W2, Z2, X, Y) #ensuite en arriere 
        W1, B1, W2, B2 = update_params(W1, B1, dW1, dB1, W2, B2, dW2, dB2, alpha) #enfin on reinitialise les parametres de bases, mais avec les corrections appliquees
        if i % 50 == 0: #toutes les 50 iterations
            print('iterations -> ', i)
            print('accuracy -> ', get_accuracy(get_predictions(A2), Y))
    return W1, B1, W2, B2 # renvoie des parametres apres l'entrainement

# Conversion des labels en entiers
Y_train = Y_train.astype(int)
print("Valeurs uniques dans Y_train:", np.unique(Y_train))

W1, B1, W2, B2 = gradient_descent(X_train, Y_train, 500, 0.1) #entrainement du reseau de neurones

