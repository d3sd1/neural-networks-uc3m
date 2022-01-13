from scipy.io import arff
import pandas as pd
import numpy as np 
data = arff.loadarff('concrete.dat')

df = pd.DataFrame(data[0])
normalized_df=(df-df.min())/(df.max()-df.min())
train, validate, test = np.split(normalized_df.sample(frac=1, random_state=42), [int(.7*len(df)), int(.85*len(df))])# 70% train 15% val 15% test .85 = 1 - 15%

train.to_csv('data/train.csv', encoding='utf-8',index=False)
validate.to_csv('data/validate.csv', encoding='utf-8',index=False)
test.to_csv('data/test.csv', encoding='utf-8',index=False)