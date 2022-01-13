import numpy as np
import pandas as pd
import csv

class Adaline(object):
	#Declaramos el constructor del adaline el cual tiene valores por defecto
	def __init__(self, nIterations=200, rate=0.01) -> None:
		self.nIterations = nIterations
		self.rate = rate

	#Calcula la salida de la red, multipilcando el vector de pesos por la matriz de datos y sumando el umbral
	def netOutput(self, X):
		return np.dot(X,self.weight) + self.threshold

	def fit(self, X_train, y_train, X_val, y_val):

		# Initial random weights and 
		#Asignamos los pesos y el umbral de forma aleatoria, lo hacemos aqui y no en el init para que se restablezcan en cada entreno
		self.weight = np.random.random_sample(X_train.shape[1])
		self.threshold = np.random.random_sample(1)

		#Iteramos tantas veces como se indico en el constructor
		for i in range(self.nIterations):
			#Calculamos la salida de la red y el error en este ciclo
			output_Train = self.netOutput(X_train)
			errors_Train = y_train - output_Train

			#Calculamos el umbral y los nuevo pesos y el nuevo error cuadratico medio de los datos de entrenamiento
			self.weight += self.rate * X_train.T.dot(errors_Train)
			self.threshold += self.rate * errors_Train.sum()
			mseIter_Train = (errors_Train**2).sum() /X_train.shape[0]

			#Calculamos los valores y el error del dataset de validacion
			output_Val = self.netOutput(X_Val)
			errors_Val = y_val - output_Val
			mseIter_Val = (errors_Val**2).sum() /X_val.shape[0]

			#Hacemos print con la iteracion y ambos errores
			print("Epoch: " + str(i) + " mse train " + str(mseIter_Train) + " mse val " + str(mseIter_Val))
			
		return self
	
	#Definimos la funcion predict para que sea mas legible para el usuario aunque es igual que netOutput
	def predict(self, X):
		return self.netOutput(X)

	
#Preparamos el dataset de entrenamiento
df_Train = pd.read_csv('data/train.csv')
X_Train = df_Train.iloc[:, 0:7].values
y_Train = df_Train.iloc[:, 8].values

#Preparamos el dataset de validacion
df_Val = pd.read_csv('data/validate.csv')
X_Val = df_Val.iloc[:, 0:7].values
y_Val = df_Val.iloc[:, 8].values

#Preparamos el dataset de test
df_Test = pd.read_csv('data/test.csv')
X_Test = df_Test.iloc[:, 0:7].values
y_Test = df_Test.iloc[:, 8].values

#Creamos el adaline y lo entrenamos
adaline = Adaline(500, 0.001).fit(X_Train,y_Train,X_Val,y_Val)

#Calculamos las predicciones para el dataset de test y lo mostramos en pantalla
predictions = adaline.predict(X_Test)
errorPredict = y_Test - predictions
print( "mse test: " + str((errorPredict**2).sum() /X_Test.shape[0]))

#Guardamos las predicciones en un csv
np.savetxt('output.csv',predictions)

#Guardamos los pesos y el umbral en un json
f = open("weights.json", "w")
f.write("{\n\t\"threshold\": " + str(adaline.threshold[0])+",")
for i in range(len(adaline.weight)):
	 f.write("\n\t\"w" + str(i) + "\": " + str(adaline.weight[i]))
	 if(i != len(adaline.weight)-1):
		 f.write(",")
f.write("\n}")
f.close()