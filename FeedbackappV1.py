from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from joblib import load
import pandas as pd
import lime
import lime.lime_tabular
import os
import shap



FeedbackappV2 = Flask(__name__)


# Set the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjusted file paths
model_path = os.path.join(BASE_DIR, 'rf_model_8020_sept22.joblib')
scaler_path = os.path.join(BASE_DIR, 'scaler_rf8020_sept22.joblib')
selector_path = os.path.join(BASE_DIR, 'selector_rf8020sept22.joblib')
#config_path = os.path.join(BASE_DIR, 'configpred50.yaml')
training_data_path = os.path.join(BASE_DIR, 'training_rf8020_sept22.csv')

# Load configuration safely
# Load trained model, scaler, and feature selector
try:
    model1 = load(model_path)
    scaler = load(scaler_path)
    selector = load(selector_path)
except Exception as e:
    print(f"Error loading the model/scaler/selector files: {str(e)}")

# Load trained model, scaler, and feature selector safely
try:
    model1 = load(model_path)
    scaler = load(scaler_path)
    selector = load(selector_path)
except Exception as e:
    print(f"Error loading the model/scaler/selector files: {str(e)}")




feature_mapping = {
    'age': 'Age of company in years',
    'internetActivityScore': 'Internet Activity Score',
    'renowned': 'How renowned your founders are in professional circles',
    'googlePageRank': "the Google page rank of company's website",
    'percentSkillDataScience': 'The team percentage skill in Data Science',
    'lastFundingAmount' : 'Last Funding Amount',
    'verticals': ('Catering to product/service across verticals',),
    'employeeCount': 'The employee head count',
    'employeePerYear':"The number of employees per year of company's existence",
    'structuredDataOnly': ('The focus on structured or unstructured data',),
    'bigDataBusiness': ('Big Data Business',),
    'hyperLocalisation': ('Hyper localisation',),
    'relevanceExperience': ("The relevance of experience to the company's venture",), 
    'exposureGlobe': ('The exposure across the globe of the team',),      
    'controversialHistory': ("The controversy about your founder or co founder's history",),
    'legalriskIP':('The legal risk and intellectual property',),   
    'globalIncubation': ('the invested through global incubation competitions',),  
    'survivalRecession': ("How the Recession significantly impacted the company's viability",'The chances of Survival during the recession when applicable to the company',),  
    'technicalProficiency': ('Technical proficiencies to analyse and interpret unstructured data',)  
}  

reverse_feature_mapping = {v: k for k, vs in feature_mapping.items() for v in (vs if isinstance(vs, tuple) else (vs,))}

# Extracting the original feature names
original_feature_names = []
for original_features in feature_mapping.values():
    if isinstance(original_features, tuple):
        original_feature_names.extend(original_features)
    else:
        original_feature_names.append(original_features)

# Load training data as DataFrame
training_data_df = pd.read_csv(training_data_path)
training_data = training_data_df.values


# Initialize LIME with selected features
#explainer = lime.lime_tabular.LimeTabularExplainer(training_data=training_data, feature_names=original_feature_names, class_names=['Negative', 'Positive'], verbose=True, mode='classification')

best_model = model1.best_estimator_
explainer = shap.Explainer(best_model.predict, training_data)

non_binary_features = ['Internet Activity Score', 'The employee head count', "the Google page rank of company's website", 'Age of company in years', 
                       "The number of employees per year of company's existence", 'The team percentage skill in Data Science', 'How renowned your founders are in professional circles', 
                       'Last Funding Amount']

#[    'Age of company in years', 'Internet Activity Score', 'Renowned in professional circle', 
    #'google page rank of company website', 'Percent_skill_Data Science', 'Employee Count', 'Employees per year of company existence', 'Last Funding Amount']

def preprocess_input(data):
    try:
        print("Input Data:", data)
        df = pd.DataFrame(data, index=[0])
        
        preprocessed_data = {}
        for html_feature, original_features in feature_mapping.items():
            value = df[html_feature].values[0]
            print(f"Processing HTML Feature: {html_feature}, Value: {value}")  # Debug print

            if isinstance(original_features, tuple):  # Handling combined features
                                
                if html_feature in ["hyperLocalisation", "structuredDataOnly","legalriskIP", "controversialHistory", "globalIncubation", "bigDataBusiness"]:
                    # Assuming "No" is represented by 1 and "Yes" by 0 in your dataset for these features
                    preprocessed_data[original_features[0]] = 1 if value == "No" else 0             
                    
                elif html_feature in ["verticals", "relevanceExperience", "exposureGlobe", "technicalProficiency"]:
                    # Assuming "Yes" is represented by 1 and "No" by 0 in your dataset for these features
                    preprocessed_data[original_features[0]] = 1 if value == "Yes" else 0

                elif html_feature == "survivalRecession":
                    if value == "Yes":
                        preprocessed_data[original_features[0]] = 0
                        preprocessed_data[original_features[1]] = 0
                    elif value == "No":
                        preprocessed_data[original_features[0]] = 1
                        preprocessed_data[original_features[1]] = 0
                    elif  value== "Not Applicable":# Handling "Not Applicable" case
                        preprocessed_data[original_features[0]] = 0
                        preprocessed_data[original_features[1]] = 1
                    

            else:  # Handling non-combined features
                print(f"Adding Non-Combined Feature: {original_features}, Value: {value}")  # Debug print
                # Convert value to float if it is a string
                if isinstance(value, str):
                    value = int(value)
                preprocessed_data[original_features] = value

        # Create a DataFrame from the preprocessed data
        preprocessed_df = pd.DataFrame([preprocessed_data])
 	# Scale only the non-binary features
        preprocessed_df[non_binary_features] = scaler.transform(preprocessed_df[non_binary_features])
        #preprocessed_df[selector] = scaler.transform(preprocessed_df[selector])

        print(preprocessed_df.columns)
        print(training_data_df.columns)



	# Safety check: Ensure the preprocessed DataFrame has the same columns as the training data DataFrame
       # if set(preprocessed_df.columns) != set(training_data_df.columns):
         #   raise ValueError("Mismatch in columns between input data and training data!")
            
        return preprocessed_df

    except Exception as e:
        raise ValueError(f"Error in preprocessing input data: {str(e)}")

                   
@FeedbackappV2.route('/', methods=['GET'])
def home():
    return render_template('FeedbackappV2.html')

@FeedbackappV2.route('/Feedback/appV2', methods=['POST'])
def predict_app():
    
    
    # Get form data
    data = request.form.to_dict()
    print("Form Data:", data)  # Debugging line

    # Check for empty fields in form data
    for key, value in data.items():
        if not value:
            return jsonify({'error': f'The field {key} is empty. Please fill it before submitting.'})

    
    # Preprocess data
    input_data = preprocess_input(data)
    print("Preprocessed Input Data:")
    print(input_data)
    

    if input_data.empty:  # Handle case when input data becomes None after preprocessing
        return jsonify({'error': 'Data could not be processed'})

    # Make a prediction 
    prediction = model1.predict(input_data)

    # Get the probability of success
    probabilities = model1.predict_proba(input_data)
    print("Prediction:", prediction)  # Debugging line
    print("Probabilities:", probabilities)  # Debugging line


    #success_probability = float(probabilities[0][1])
    success_probability = probabilities[0][0].item() if prediction[0] == 0 else probabilities[0][1].item()

    # Convert probability to percentage
    success_probability *= 100
     # Get the explanation for the prediction using SHAP
    shap_values = explainer(input_data)  # Note: We use the explainer like a function to get the SHAP values

    # Get the names of the features
    feature_names = input_data.columns.tolist()

    # Prepare the explanation data
    explanation = []
    for i, feature_name in enumerate(feature_names):
        explanation.append({
            "feature": feature_name,
            "weight": shap_values.values[0][i]  # Adjusted to get the values from the SHAP explanation object
        })

    # Formatting and sorting the explanation
    formatted_explanation = []
    for exp in explanation:
        formatted_feature = " ".join(word.capitalize() for word in exp["feature"].split("_"))
        formatted_explanation.append({"feature": formatted_feature, "weight": exp["weight"]})

    # Sorting the explanation by absolute weight value
    sorted_explanation = sorted(formatted_explanation, key=lambda x: abs(x["weight"]), reverse=True)

    # Mapping prediction to 'success' or 'failure'
    result = 'Success' if prediction[0] == 1 else 'Failure'

    # Send the prediction, probability, and explanation back as a JSON response
    return render_template('Feedback_template.html', 
                           prediction=result, 
                           probability=success_probability, 
                           explanation=sorted_explanation, 
                           data=data, 
                           feature_mapping=feature_mapping)

# Run the flask app
if __name__ == '__main__':
    FeedbackappV2.run(debug=True)
  