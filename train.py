import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Shared training data: [sqft, bedrooms, age]
X_train = [
    [2000, 3, 10], [1200, 2, 25], [2500, 4, 5], 
    [800, 1, 30], [1800, 3, 12], [950, 2, 20]
]

# --- 1. Train Classifier (High/Low Value) ---
y_train_class = [1, 0, 1, 0, 1, 0]
clf_model = RandomForestClassifier(random_state=42)
clf_model.fit(X_train, y_train_class)
joblib.dump(clf_model, "classifier_model.joblib")
print("Saved classifier_model.joblib")

# --- 2. Train Regressor (Actual Price) ---
y_train_price = [450000, 250000, 600000, 150000, 400000, 200000]
reg_model = RandomForestRegressor(random_state=42)
reg_model.fit(X_train, y_train_price)
joblib.dump(reg_model, "regressor_model.joblib")
print("Saved regressor_model.joblib")