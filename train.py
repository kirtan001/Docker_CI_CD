import pandas as pd 
import pickle 
from sklearn.linear_model import LogisticRegression 
URL = "D:\\1_sem_4\\f_mlops\\docker_ci_cd\\data\\winequality-red.csv" 
df = pd.read_csv(URL, sep=";") 
df["quality"] = (df["quality"] >= 6).astype(int) 
X = df.drop("quality", axis=1) 
y = df["quality"] 
model = LogisticRegression(max_iter=1000) 
model.fit(X, y) 
with open("model.pkl", "wb") as f: 
    pickle.dump(model, f) 
print("Model trained using UCI data") 