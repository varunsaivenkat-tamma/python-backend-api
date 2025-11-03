from flask import Flask, request, jsonify
import pandas as pd
import pickle
from flask_cors import CORS

# =============================
# 1️⃣ Setup Flask App
# =============================
app = Flask(__name__)
CORS(app)

# =============================
# 2️⃣ Load Model and Metadata
# =============================
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("model/features.pkl", "rb") as f:
    feature_cols = pickle.load(f)

# Load dataset for recommendations
df = pd.read_csv("dataset/enriched_travel_package_dataset.csv")
df.columns = df.columns.str.strip()

# =============================
# 3️⃣ /predict Endpoint
# =============================
@app.route("/predict", methods=["POST"])
def predict_cost():
    try:
        data = request.get_json()

        # Extract user input
        source = data["Source"]
        destination = data["Destination"]
        destination_type = data["Destination_Type"]
        season = data["Season"]
        package_type = data["Package_Type"]
        package_category = data["Package_Category"]
        duration = float(data["Duration_Days"])
        user_budget = float(data["User_Budget"])
        user_rating = float(data["User_Rating"])

        # Encode using the same label encoders
        def safe_encode(col, value):
            le = label_encoders[col]
            if value not in le.classes_:
                return 0  # fallback if unseen
            return le.transform([value])[0]

        input_data = {
            "Source_encoded": safe_encode("Source", source),
            "Destination_encoded": safe_encode("Destination", destination),
            "Destination_Type_encoded": safe_encode("Destination_Type", destination_type),
            "Season_encoded": safe_encode("Season", season),
            "Duration_Days": duration,
            "User_Budget": user_budget,
            "User_Rating": user_rating,
            "Package_Type_encoded": safe_encode("Package_Type", package_type),
            "Package_Category_encoded": safe_encode("Package_Category", package_category),
        }

        # Convert to DataFrame and scale
        df_input = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df_input[feature_cols])

        # Predict
        predicted_cost = model.predict(df_scaled)[0]

        return jsonify({"predicted_cost": round(float(predicted_cost), 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# =============================
# 4️⃣ /recommend Endpoint
# =============================
@app.route("/recommend", methods=["POST"])
def recommend_packages():
    try:
        data = request.get_json()

        source = data.get("Source", "").strip().title()
        destination = data.get("Destination", "").strip().title()
        destination_type = data.get("Destination_Type", "").strip().title()
        season = data.get("Season", "").strip().title()
        duration = float(data.get("Duration_Days", 0))
        budget = float(data.get("User_Budget", 0))

        # Case-insensitive filtering with relaxed conditions
        filtered = df[
            (df["Source"].str.title() == source)
            & (df["Destination"].str.title() == destination)
            & (df["Destination_Type"].str.title() == destination_type)
            & (df["Season"].str.title() == season)
        ]

        # Relax duration/budget filters
        filtered = filtered[
            (filtered["Duration_Days"].between(duration - 3, duration + 3))
            & (filtered["Package_Price"].between(budget * 0.6, budget * 1.4))
        ]

        if filtered.empty:
            # fallback: return top-rated similar destination
            filtered = (
                df[df["Destination"].str.title() == destination]
                .sort_values(by=["User_Rating", "Package_Price"], ascending=[False, True])
                .head(3)
            )

        results = filtered[
            [
                "Trip_ID",
                "Package_Title",
                "Source",
                "Destination",
                "Destination_Type",
                "Duration_Days",
                "Season",
                "Cost",
                "User_Budget",
                "User_Rating",
                "Package_Type",
                "Package_Price",
                "Package_Category",
                "Image_URL",
                "Trip_Description",
                "Itinerary_Plan",
                "Highlights",
            ]
        ].to_dict(orient="records")

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# =============================
# 5️⃣ Run Server
# =============================
if __name__ == "__main__":
    app.run(debug=True)
