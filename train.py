import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import pickle
import os

# =============================
# 1️⃣ Load Dataset
# =============================
df = pd.read_csv("dataset/enriched_travel_package_dataset.csv")
df.columns = df.columns.str.strip()

print("✅ Columns:", df.columns.tolist())

# =============================
# 2️⃣ Encode Categorical Columns
# =============================
label_encoders = {}
categorical_cols = [
    "Source",
    "Destination",
    "Destination_Type",
    "Season",
    "Package_Type",
    "Package_Category",
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# =============================
# 3️⃣ Define Features and Target
# =============================
feature_cols = [
    "Source_encoded",
    "Destination_encoded",
    "Destination_Type_encoded",
    "Duration_Days",
    "Season_encoded",
    "User_Budget",
    "User_Rating",
    "Package_Type_encoded",
    "Package_Category_encoded",
]
target_col = "Cost"

X = df[feature_cols]
y = df[target_col]

# =============================
# 4️⃣ Train/Test Split + Scale
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 5️⃣ Train Model
# =============================
model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

# =============================
# 6️⃣ Evaluate
# =============================
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(f"🎯 Model trained successfully. MAE: ₹{mae:.2f}")

# =============================
# 7️⃣ Save Artifacts
# =============================
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

# Save feature column order for prediction reference
with open("model/features.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

print("✅ Model, scaler, encoders, and feature list saved successfully.")
