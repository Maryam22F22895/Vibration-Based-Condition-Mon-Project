from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secure your app with a strong secret key

# Load Model and Label Encoder
model = tf.keras.models.load_model("cnn_model.h5")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

fault_map = {
    "N": {"status": "Healthy", "fault_type": None, "diameter": None},
    "7_OR3": {"status": "Faulty", "fault_type": "Outer Race Position 'centered'", "diameter": "0.007”"},
    "7_OR2": {"status": "Faulty", "fault_type": "Outer Race Position 'Orthogonal'", "diameter": "0.007”"},
    "7_OR1": {"status": "Faulty", "fault_type": "Outer Race Position 'Opposite'", "diameter": "0.007”"},
    "7_BA": {"status": "Faulty", "fault_type": "Ball", "diameter": "0.007”"},
    "7_IR": {"status": "Faulty", "fault_type": "Inner Race", "diameter": "0.007”"},
    "21_OR3": {"status": "Faulty", "fault_type": "Outer Race Position 'centered'", "diameter": "0.021”"},
    "21_OR2": {"status": "Faulty", "fault_type": "Outer Race Position 'Orthogonal'", "diameter": "0.021”"},
    "21_OR1": {"status": "Faulty", "fault_type": "Outer Race Position 'Opposite'", "diameter": "0.021”"},
    "21_IR": {"status": "Faulty", "fault_type": "Inner Race", "diameter": "0.021”"},
    "21_BA": {"status": "Faulty", "fault_type": "Ball", "diameter": "0.021”"},
    "14_OR1": {"status": "Faulty", "fault_type": "Outer Race Position 'Opposite'", "diameter": "0.014”"},
    "14_IR": {"status": "Faulty", "fault_type": "Inner Race", "diameter": "0.014”"},
    "14_BA": {"status": "Faulty", "fault_type": "Ball", "diameter": "0.014”"}
}


# Hardcoded user database for simplicity
users = {
    "employee1": generate_password_hash("password1"),
    "employee2": generate_password_hash("password2"),
    "employee3": generate_password_hash("password3"),
    "employee4": generate_password_hash("password4"),
    "employee5": generate_password_hash("password5"),
    "employee6": generate_password_hash("password6"),
    "employee7": generate_password_hash("password7"),
    "employee8": generate_password_hash("password8"),
    "employee9": generate_password_hash("password9"),
    "employee10": generate_password_hash("password10"),
}

# Splash Screen
@app.route('/')
def splash():
    return render_template("splash.html")
#login 
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in users and check_password_hash(users[username], password):
            session['user'] = username
            return redirect(url_for('main_page'))
        else:
            error_message = "Invalid username or password. Please try again."
            return render_template("login.html", error_message=error_message)

    return render_template("login.html")

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# File Upload and Model Prediction Page
@app.route('/main', methods=["GET", "POST"])
def main_page():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            return "No file provided!", 400

        # Ensure the 'uploads' directory exists
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)

        filename = secure_filename(file.filename)
        file_path = os.path.join(uploads_dir, filename)
        file.save(file_path)

        try:
            if file.filename.endswith(".csv"):
                data = pd.read_csv(file_path)
                win_len = 500
                stride = 300
                X = []

                for i in np.arange(0, len(data) - win_len, stride):
                    temp = data.iloc[i:i + win_len, :-1].values
                    temp = temp.reshape((1, -1, 1))
                    X.append(temp)

                predictions = model.predict(X)
                labels = encoder.inverse_transform(np.argmax(predictions, axis=1))

                # Map predictions to their descriptions
                detailed_predictions = []
                for label in labels:
                    result = fault_map.get(label)
                    if result:
                        detailed_predictions.append(f"{result['status']} - Fault Type: {result['fault_type']}, Fault Diameter: {result['diameter']}")
                    else:
                        detailed_predictions.append("Unknown fault")

                return render_template("results.html", predictions=detailed_predictions)

            # Handle .mat file processing (similar to CSV)
            elif file.filename.endswith(".mat"):
                import scipy.io
                mat = scipy.io.loadmat(file_path)

                key_name = list(mat.keys())[3]  
                DE_data = mat.get(key_name)

                if DE_data is None:
                    return "DE_data not found in the .mat file", 400

                win_len = 500
                stride = 300
                X = []

                for i in np.arange(0, len(DE_data) - win_len, stride):
                    temp = DE_data[i:i + win_len]
                    temp = temp.reshape((1, -1, 1))
                    X.append(temp)

                predictions = model.predict(X)
                labels = encoder.inverse_transform(np.argmax(predictions, axis=1))

                # Map predictions to their descriptions
                detailed_predictions = []
                for label in labels:
                    result = fault_map.get(label)
                    if result:
                        detailed_predictions.append(f"{result['status']} - Fault Type: {result['fault_type']}, Fault Diameter: {result['diameter']}")
                    else:
                        detailed_predictions.append("Unknown fault")

                return render_template("results.html", predictions=detailed_predictions)

        except Exception as e:
            return str(e), 500

    return render_template("main.html")

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
 