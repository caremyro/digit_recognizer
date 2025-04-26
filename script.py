import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('train.csv') #on recupere les donnees du fichier
l, c = data.shape 
np.random.shuffle(data)
