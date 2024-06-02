# Importing necessary libraries
import warnings
from sklearn.exceptions import DataConversionWarning
from flask import Flask, render_template, request, jsonify
import joblib as jb
import pandas as pd

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings("ignore", category=DataConversionWarning)

# Defining functions to preprocess input data
def new_age(age):
    if 25 <= age <= 35:
        return 0
    elif 35 <= age <= 50:
        return 1
    elif 50 <= age <= 90:
        return 2
    else:
        return -1
    
def gender(sex):
    if sex == 'Male':
        return 0
    elif sex == 'Female':
        return 1
    else:
        return -1

def cohort(cohort_type):
    if cohort_type == 'Cohort1':
        return 0
    elif cohort_type == 'Cohort2':
        return 1
    else:
        return -1

def origin(sample_origin):
    if sample_origin == 'BPTB':
        return 0
    else:
        return 1

def cr(creatinine):
    if creatinine <= 0.37320:
        return 0
    elif 0.38 <= creatinine <= 1.139:
        return 1
    elif creatinine > 1.14:
        return 2
    else:
        return -1

# Loading the trained model
model_file_path = "D:/pancreas/gradient_boosting_model.pkl"
with open(model_file_path, 'rb') as f:
    gb_model = jb.load(f)

# Initializing Flask app
app = Flask(__name__)

# Defining routes
@app.route('/')
def home():
    return render_template('pan.html')  # Render HTML template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form submission
        age = float(request.form['age'])
        sex = request.form['sex']
        cohort_type = request.form['cohort_type']
        sample_origin = request.form['sample_origin']
        creatinine = float(request.form['creatinine'])
        LYVE1 = float(request.form['LYVE1'])
        REG1B = float(request.form['REG1B'])

        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cohort_type': [cohort_type],
            'sample_origin': [sample_origin],
            'creatinine': [creatinine],
            'LYVE1': [LYVE1],
            'REG1B': [REG1B]
        })

        # Preprocessing input data
        input_data['age'] = input_data['age'].apply(new_age)
        input_data['sex'] = input_data['sex'].apply(gender)
        input_data['cohort_type'] = input_data['cohort_type'].apply(cohort)
        input_data['sample_origin'] = input_data['sample_origin'].apply(origin)
        input_data['creatinine'] = input_data['creatinine'].apply(cr)

        # Make prediction
        prediction = gb_model.predict(input_data)

        # Define result message based on prediction
        if prediction[0] == 3:
            result_message = 'The patient has pancreatic cancer.'
        elif prediction[0] == 2:
            result_message = 'The patient has a non-cancerous pancreatic condition.'
        elif prediction[0] == 1:
            result_message = 'The patient is healthy.'
        else:
            result_message = 'Unknown diagnosis.'

        # Return prediction result as JSON
        return jsonify({'prediction': prediction[0], 'result': result_message})

    except Exception as e:
        # Return error message if an exception occurs
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
