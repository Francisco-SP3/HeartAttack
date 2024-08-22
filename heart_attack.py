# Francisco Salas Porras A01177893
# Heart Attack Treatment
# Dataset: https://www.kaggle.com/datasets/waqi786/heart-attack-dataset?resource=download

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('heart_attack_dataset.csv')
df.head()

# Analyze dataset
df.isnull().sum()