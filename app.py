from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Check if files exist before loading
def load_pkl_file(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        print(f"⚠️ Warning: {file_name} not found!")
        return None

model = load_pkl_file("random_forest_model.pkl")
encoder = load_pkl_file("onehot_encoder.pkl")
scaler = load_pkl_file("scaler.pkl")  # Optional
label_encoder = load_pkl_file("label_encoder.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    recommended_plant = None

    if request.method == 'POST':
        try:
            if not all([model, encoder, label_encoder]):
                return render_template("index.html", error="Model files are missing! Please retrain and save them properly.")

            # Get form data
            age = float(request.form["age"])
            gender = request.form["gender"]
            problem = request.form["problem"]
            allergy = request.form["allergy"]
            diet = request.form["diet"]
            history = request.form["history"]
            medications = request.form["medications"]

            # Prepare input data
            input_data = np.array([[age, gender, problem, allergy, diet, history, medications]])

            # Encode categorical features
            encoded_data = encoder.transform(input_data[:, 1:])  # Skip age for encoding

            # Scale numerical data (age)
            if scaler:
                numerical_data = scaler.transform(input_data[:, :1].astype(float))
            else:
                numerical_data = input_data[:, :1].astype(float)  # Use raw age data if no scaler

            processed_data = np.hstack((numerical_data, encoded_data))

            # Predict
            prediction = model.predict(processed_data)
            recommended_plant = label_encoder.inverse_transform(prediction)[0]

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html", recommended_plant=recommended_plant)

if __name__ == "__main__":
    app.run(debug=True)
