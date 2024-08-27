# Francisco Salas Porras A01177893
# Clasificación de números escritos a mano

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split

###################################### Hiperparámetros ######################################

# Seed para reproducibilidad
np.random.seed(0)

# Hiperparámetros para entrenamiento (alpha, epochs)
hiper = []
hiper.append( (1, 10000) ) # 0 
hiper.append( (4, 3000) ) # 1 
hiper.append( (1, 10000) ) # 2 
hiper.append( (1, 3000) ) # 3 
hiper.append( (1, 6000) ) # 4 
hiper.append( (1, 3000) ) # 5 
hiper.append( (1, 5000) ) # 6 
hiper.append( (1, 5000) ) # 7 
hiper.append( (1.5, 2000) ) # 8
hiper.append( (1, 3000) ) # 9

###################################### Definición de funciones ######################################

# Función de hipótesis 
def sig_function(X):
  return 1/(1+math.e**(-X))

# Función de costo (Negative log loss)
def NLL_function(yP, y):
  return -np.sum(np.dot(y.T, np.log(yP) + np.dot((1-y).T, np.log(1-yP)))) / len(yP)

# Función de entrenamiento (Logistic Regression)
def logReg(X, y, alpha, epochs):
  
  # Datapoints | (1797,1)
  y_ = np.reshape(y, (len(y), 1))

  # Número de parámetros
  N = len(X)

  # Inicialización aleatoria de los parámetros
  theta = np.random.randn(len(X[0]) + 1, 1)

  # Agregar columna de bias
  X = np.c_[np.ones((len(X), 1)), X]

  # Memoria de costos por epoch
  avg_loss_list = []

  for epoch in range(epochs+1):

    # Predicción / Evaluación de función de hipótesis (Sigmoidal) | (1797,65).(65,1) = (1797,1)
    sigmoid_x_theta = sig_function(X @ theta)

    # Optimización / Gradiente de los parámetros (Gradient descent) | (65,1797).(1797,1) = (65, 1)
    grad = (1/N) * X.T @ (sigmoid_x_theta - y_)

    # Actualización / Cambio de los parámetros según el aprendizaje (Thetas)
    theta = theta - (alpha * grad)

    # Resultado / Nueva evaluación de hipótesis con parámetros actualizados | (1797,65).(65,1) = (1797,1)
    yP = sig_function(X @ theta)

    # Error / Evaluación de función de costo (Negative log loss)
    avg_loss = NLL_function(yP, y_)

    # Imprimir progreso del modelo
    if epoch % 1000 == 0:
      print('epoch: {} | avg_loss: {}'.format(epoch, avg_loss))

    # Guardar promedio de costo por epoch
    avg_loss_list.append(avg_loss)
    
  return theta, avg_loss_list

# Función de prueba (Logistic Regression)
def testLogReg(X, y, theta):
  
  # Agregar columna de bias
  X = np.c_[np.ones((len(X), 1)), X]

  # Datapoints | (1797,1)
  y_ = np.reshape(y, (len(y), 1))

  # Predicción / Evaluación de modelo entrenado | (1797,65).(65,1) = (1797,1)
  yP = sig_function(X @ theta)

  # Error / Evaluación de función de costo (Negative log loss)
  avg_loss = NLL_function(yP, y_)

  # Matriz de confusión
  mat = np.zeros((2, 2))
  for i in range(len(yP)):
    if yP[i] >= 0.5:
      if y_[i] == 1:
        mat[0][0] += 1
      else:
        mat[1][0] += 1
    else:
      if y_[i] == 1:
        mat[0][1] += 1
      else:
        mat[1][1] += 1

  # Precision, Recall, Accuracy
  precision = mat[0][0] / (mat[0][0] + mat[0][1])
  recall = mat[0][0] / (mat[0][0] + mat[1][0])
  accuracy = (mat[0][0] + mat[1][1]) / np.sum(mat)

  # Imprimir resultados
  print('Test loss:', avg_loss)
  print('Precision: ', precision)
  print('Recall: ', recall)
  print('Accuracy: ', accuracy)
  print('Confusion matrix: \n', mat)

  return avg_loss, precision, recall, accuracy

# Función de clasificación múltiple (One vs All)
def oneVsAll(X_train, X_test, y_train, y_test, hiper):
  
  # Número de clases (0 - 9)
  classes = np.unique(y)

  # Parámetros de cada clase
  thetas = []

  # Entrenamiento de cada clase
  for i in range(len(classes)):
    print('------------------------------------------')
    print('Training model for number: {}'.format(i))
    print('Alpha: {} | Epochs: {}'.format(hiper[i][0], hiper[i][1]))
    print('------------------------------------------')

    # Extracción de target 
    y_tr = (y_train == i).astype(int)
    y_te = (y_test == i).astype(int)

    # Entrenamiento del modelo
    theta, trainLoss = logReg(X_train, y_tr, hiper[i][0], hiper[i][1])

    # Prueba del modelo
    testLoss, precision, recall, accuracy = testLogReg(X_test, y_te, theta)

    # Guardar parámetros
    thetas.append(theta)
    
  return thetas

# Función de predicción
def predict(X_test, y_test, thetas):
  
    # Agregar columna de bias
    #X_test = X_test.reshape(-1, 1)
    #X_test = np.vstack([np.ones((1, 1)), X_test])
    #X_test = np.concatenate(([1], X_test))
    X_test = np.c_[np.ones((len(X_test), 1)), X_test] # add x0 for bias
  
    # Predicción de cada clase
    yP = []
    for i in range(10):
      yP.append(sig_function(X_test.dot(thetas[i]))[0][0])
      print('Model prediction for number {}: {}'.format(i, round(yP[i],4)))

    # Predicción final
    yP = yP.index(max(yP))

    # Imprimir resultados
    print('------------------------------------------')
    print('Model prediction: ', yP)
    print('Actual target: ', y_test)
    print('------------------------------------------')
    print('\n')
  
    return yP

###################################### Entrenamiento del modelo ######################################

# Dataset de números escritos a mano
digits = datasets.load_digits()

# Extracción de features y target
X = digits["data"]
y = digits["target"]

# División de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalización de datos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entrenamiento de modelo
print('\n\n Model Training: \n\n')
thetas = oneVsAll(X_train, X_test, y_train, y_test, hiper)

###################################### Resultados del modelo ######################################

# Predicción de modelo
print('\n\n Model Predictions: \n\n')
predict([list(X_test[0])], y_test[0], thetas)
predict([list(X_test[1])], y_test[1], thetas)
predict([list(X_test[2])], y_test[2], thetas)
predict([list(X_test[3])], y_test[3], thetas)
predict([list(X_test[4])], y_test[4], thetas)
predict([list(X_test[5])], y_test[5], thetas)