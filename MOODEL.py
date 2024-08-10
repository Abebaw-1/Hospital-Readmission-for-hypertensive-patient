import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
df = pd.read_csv("C:/Users/abebaw/Final Thesis/MOODDEL.csv")

Xx = df.iloc[:, :-1]
Yy = df.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(Xx, Yy, stratify=Yy, test_size=0.20)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(Xx, Yy, stratify=Yy, test_size=0.30)

import xgboost as xgb
from sklearn.model_selection import cross_val_score

model_1 = xgb.XGBClassifier(n_estimators = 10, learning_rate = 0.2,max_depth=10)
model_1.fit(X_train_1, y_train_1)
# Make pickle file of our model
import pickle
pickle.dump(model_1, open("xgb_MOODEL.pkl", "wb"))
with open('xgb_MOODEL.pkl', 'rb') as f:
    xgb_model = pickle.load(f)