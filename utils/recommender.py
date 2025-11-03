import pandas as pd
import joblib
import os

class TravelRecommender:
    def __init__(self):
        self.model = joblib.load(os.path.join('model', 'cost_model.joblib'))
        self.scaler = joblib.load(os.path.join('model', 'scaler.joblib'))
        self.le_destination = joblib.load(os.path.join('model', 'destination_encoder.joblib'))
        self.le_accommodation = joblib.load(os.path.join('model', 'accommodation_encoder.joblib'))
        self.le_transportation = joblib.load(os.path.join('model', 'transportation_encoder.joblib'))
        self.data = pd.read_csv(os.path.join('dataset', 'enriched_travel_package_dataset.csv'))

    def estimate_cost(self, destination, duration, travelers, accommodation_type, transportation_type):
        # Encode inputs
        dest_encoded = self.le_destination.transform([destination])[0]
        acc_encoded = self.le_accommodation.transform([accommodation_type])[0]
        trans_encoded = self.le_transportation.transform([transportation_type])[0]

        # Prepare input
        input_data = [[dest_encoded, duration, travelers, acc_encoded, trans_encoded]]
        input_scaled = self.scaler.transform(input_data)

        # Predict cost
        estimated_cost = self.model.predict(input_scaled)[0]
        return estimated_cost

    def get_recommendations(self, budget, duration, travelers, preferred_destination=None):
        # Filter packages within budget
        suitable_packages = self.data[
            (self.data['Cost'] <= budget) &
            (self.data['Duration'] >= duration - 2) &  # Allow some flexibility
            (self.data['Duration'] <= duration + 2) &
            (self.data['Travelers'] == travelers)
        ]

        if preferred_destination:
            suitable_packages = suitable_packages[suitable_packages['Destination'] == preferred_destination]

        # Sort by cost (ascending) and return top recommendations
        recommendations = suitable_packages.sort_values('Cost').head(5)
        return recommendations.to_dict('records')
