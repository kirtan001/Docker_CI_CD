import pandas as pd 
import pickle 
from sklearn.linear_model import LogisticRegression 
DATA_PATH = "data/winequality-red.csv"
df = pd.read_csv(DATA_PATH, sep=";")
df["quality"] = (df["quality"] >= 6).astype(int) 
X = df.drop("quality", axis=1) 
y = df["quality"] 
model = LogisticRegression(max_iter=1000) 
model.fit(X, y) 
with open("model.pkl", "wb") as f: 
    pickle.dump(model, f) 
print("Model trained using UCI data") 