
from flask import Flask, request, render_template
import numpy as np
import pickle
import os

# --- Loading Pickled Files ---
try:
    model = pickle.load(open('model.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    print("Model and MinMaxScaler loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading pickle file: {e}")
    print("Please ensure 'model.pkl' and 'minmaxscaler.pkl' are in the same directory as app.py")
    # You might want to exit or raise an exception here in a real application
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading pickle files: {e}")
    exit()

# --- Creating Flask App ---
# Use os.path.dirname to handle paths correctly regardless of execution directory
base_dir = os.path.dirname(os.path.abspath(__file__))
template_folder = os.path.join(base_dir, 'templates')
static_folder = os.path.join(base_dir, 'static')

# Ensure the static folder exists before initializing Flask with it
if not os.path.exists(static_folder):
    os.makedirs(static_folder)
    print(f"Created static folder at: {static_folder}")
if not os.path.exists(os.path.join(static_folder, 'images')):
     os.makedirs(os.path.join(static_folder, 'images'), exist_ok=True)
     print(f"Created images sub-folder within static folder.")

app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)

# --- Flask Routes ---
@app.route('/')
def index():
    # Renders the main form page
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Check if request method is POST
    if request.method == 'POST':
        result = "" # Initialize result variable
        try:
            # Get data from form and convert to float
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosporus'])
            K = float(request.form['Potassium'])
            temp = float(request.form['Temperature'])
            humidity = float(request.form['Humidity'])
            ph = float(request.form['Ph'])
            rainfall = float(request.form['Rainfall'])

            # Create feature list and convert to numpy array for scaling/prediction
            feature_list = [N, P, K, temp, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Apply the MinMaxScaler ONLY
            scaled_features = ms.transform(single_pred)

            # Make prediction using the model
            # Note: We use scaled_features directly, not final_features
            prediction = model.predict(scaled_features)

            # --- Crop Dictionary ---
            # Maps the predicted number back to the crop name
            crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                         8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                         14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                         19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

            # --- Format Output ---
            if prediction[0] in crop_dict:
                crop = crop_dict[prediction[0]]
                result = "Prediction: {} is the best crop to cultivate in these conditions.".format(crop)
            else:
                result = "Prediction: Sorry, the model returned an unknown crop code ({}).".format(prediction[0])

        except ValueError:
            # Handle cases where user input cannot be converted to float
            result = "Error: Please enter valid numbers for all input fields."
        except Exception as e:
            # Handle other potential errors during prediction/scaling
            print(f"Error during prediction: {e}") # Log the error for debugging
            result = "Error: Could not process the prediction. Please check server logs."

        # Render the same template, passing the result back
        return render_template('index.html', result=result)

    else:
        # If someone tries to access /predict via GET, just show the form
        return render_template('index.html')


# --- Python Main Execution ---
if __name__ == "__main__":
    # The directory checks are now done before Flask initialization
    # which is slightly cleaner, but the previous way was also fine.
    print("Starting Flask development server...")
    app.run(debug=True) # debug=True allows auto-reloading on code changes