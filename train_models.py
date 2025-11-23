# train_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

np.random.seed(42)

# Create synthetic dataset
n = 5000
data = pd.DataFrame({
    "engagement": np.random.randint(0, 100, n),
    "visits": np.random.randint(0, 50, n),
    "purchase_value": np.random.randint(5000, 150000, n),
    "purchase_freq": np.random.randint(1, 12, n),
    "retention_rate": np.random.uniform(0.3, 0.95, n),
    "past_purchase": np.random.randint(0, 300000, n)
})

# Lead Score label (synthetic logic)
labels = []
for i in range(n):
    if data.loc[i, "engagement"] > 70 and data.loc[i, "visits"] > 20:
        labels.append("High")
    elif data.loc[i, "engagement"] > 40:
        labels.append("Medium")
    else:
        labels.append("Low")

data["lead_score"] = labels

# CLV formula
data["clv"] = (
    data["purchase_value"]
    * data["purchase_freq"]
    * data["retention_rate"]
) / (1 - data["retention_rate"])

# Features
X = data[[
    "engagement",
    "visits",
    "purchase_value",
    "purchase_freq",
    "retention_rate",
    "past_purchase"
]]

# Encode labels
y = data["lead_score"]
label = LabelEncoder()
y_encoded = label.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)

# Train Lead Score Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pickle.dump(clf, open("lead_model.pkl", "wb"))
pickle.dump(label, open("label_encoder.pkl", "wb"))

# Train CLV Model
# reg = RandomForestRegressor()
# reg.fit(X_train, data.loc[X_train.index, "clv"])
X_clv = data[["purchase_value", "purchase_freq", "retention_rate"]]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clv, data["clv"], test_size=0.2
)

reg = RandomForestRegressor()
reg.fit(Xc_train, yc_train)

pickle.dump(reg, open("clv_model.pkl", "wb"))

print("Models trained and saved successfully!")
