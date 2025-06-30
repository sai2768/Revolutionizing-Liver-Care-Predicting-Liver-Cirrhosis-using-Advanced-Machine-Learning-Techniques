#!/usr/bin/env python3
"""
Liver Cirrhosis Prediction Flask Web Application
===============================================

This Flask application provides a web interface for predicting liver cirrhosis
based on clinical laboratory parameters using trained machine learning models.
"""

from flask import Flask, request, render_template, jsonify, flash
import pickle
import numpy as np
import pandas as pd
from werkzeug.exceptions import BadRequest
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'liver_cirrhosis_prediction_2024'

class LiverPredictionApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
            'Aspartate_Aminotransferase', 'Total_Protiens',
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        self.load_models()
    
    def load_models(self):
        """Load the trained model and scaler"""
        try:
            # Load the trained model
            if os.path.exists('rf_acc_68.pkl'):
                with open('rf_acc_68.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Model loaded successfully")
            else:
                logger.error("Model file not found!")
                
            # Load the scaler
            if os.path.exists('normalizer.pkl'):
                with open('normalizer.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Scaler loaded successfully")
            else:
                logger.error("Scaler file not found!")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def validate_input(self, data):
        """Validate input data"""
        required_fields = self.feature_names
        
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")
            
            try:
                float(data[field])
            except ValueError:
                raise BadRequest(f"Invalid value for {field}: must be a number")
        
        # Additional validation rules
        age = float(data['Age'])
        if not (0 <= age <= 150):
            raise BadRequest("Age must be between 0 and 150")
        
        gender = int(data['Gender'])
        if gender not in [0, 1]:
            raise BadRequest("Gender must be 0 (Female) or 1 (Male)")
        
        # Validate lab values (basic ranges)
        if float(data['Total_Bilirubin']) < 0:
            raise BadRequest("Total Bilirubin cannot be negative")
        
        if float(data['Direct_Bilirubin']) < 0:
            raise BadRequest("Direct Bilirubin cannot be negative")
        
        if float(data['Alkaline_Phosphotase']) < 0:
            raise BadRequest("Alkaline Phosphotase cannot be negative")
        
        return True
    
    def make_prediction(self, input_data):
        """Make prediction using the loaded model"""
        if self.model is None or self.scaler is None:
            raise Exception("Models not loaded properly")
        
        try:
            # Convert input to numpy array
            features = np.array([[float(input_data[feature]) for feature in self.feature_names]])
            
            # Scale the features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            prediction_proba = self.model.predict_proba(features_scaled)[0]
            
            # Get confidence
            confidence = max(prediction_proba) * 100
            
            return {
                'prediction': int(prediction),
                'confidence': round(confidence, 2),
                'probability_no_cirrhosis': round(prediction_proba[0] * 100, 2),
                'probability_cirrhosis': round(prediction_proba[1] * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise Exception(f"Error making prediction: {str(e)}")

# Initialize the prediction app
prediction_app = LiverPredictionApp()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/prediction')
def prediction_page():
    """Prediction form page"""
    return render_template('forms/prediction_form.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('inner-page.html')

@app.route('/portfolio')
def portfolio():
    """Portfolio page"""
    return render_template('portfolio-details.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        input_data = {
            'Age': request.form.get('age'),
            'Gender': request.form.get('gender'),
            'Total_Bilirubin': request.form.get('total_bilirubin'),
            'Direct_Bilirubin': request.form.get('direct_bilirubin'),
            'Alkaline_Phosphotase': request.form.get('alkaline_phosphotase'),
            'Alamine_Aminotransferase': request.form.get('alamine_aminotransferase'),
            'Aspartate_Aminotransferase': request.form.get('aspartate_aminotransferase'),
            'Total_Protiens': request.form.get('total_protiens'),
            'Albumin': request.form.get('albumin'),
            'Albumin_and_Globulin_Ratio': request.form.get('albumin_globulin_ratio')
        }
        
        # Validate input
        prediction_app.validate_input(input_data)
        
        # Make prediction
        result = prediction_app.make_prediction(input_data)
        
        # Prepare result message
        if result['prediction'] == 1:
            prediction_text = "High Risk of Liver Cirrhosis"
            recommendation = "Please consult with a hepatologist immediately for further evaluation."
            risk_level = "HIGH"
            alert_class = "alert-danger"
        else:
            prediction_text = "Low Risk of Liver Cirrhosis"
            recommendation = "Continue regular health monitoring and maintain a healthy lifestyle."
            risk_level = "LOW"
            alert_class = "alert-success"
        
        return render_template('forms/result.html',
                             prediction=prediction_text,
                             confidence=result['confidence'],
                             recommendation=recommendation,
                             risk_level=risk_level,
                             alert_class=alert_class,
                             prob_no_cirrhosis=result['probability_no_cirrhosis'],
                             prob_cirrhosis=result['probability_cirrhosis'],
                             input_data=input_data)
        
    except BadRequest as e:
        flash(f"Input Error: {str(e)}", 'error')
        return render_template('forms/prediction_form.html')
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        flash(f"Prediction Error: {str(e)}", 'error')
        return render_template('forms/prediction_form.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        prediction_app.validate_input(data)
        
        # Make prediction
        result = prediction_app.make_prediction(data)
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'probabilities': {
                'no_cirrhosis': result['probability_no_cirrhosis'],
                'cirrhosis': result['probability_cirrhosis']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create dummy model files if they don't exist (for demo purposes)
    if not os.path.exists('rf_acc_68.pkl') or not os.path.exists('normalizer.pkl'):
        print("Warning: Model files not found. Please run the training script first.")
        print("Creating dummy models for demonstration...")
        
        # Create a simple dummy model for demo
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
        dummy_scaler = StandardScaler()
        
        # Create dummy data to fit
        X_dummy = np.random.randn(100, 10)
        y_dummy = np.random.randint(0, 2, 100)
        
        dummy_scaler.fit(X_dummy)
        dummy_model.fit(dummy_scaler.transform(X_dummy), y_dummy)
        
        # Save dummy models
        with open('rf_acc_68.pkl', 'wb') as f:
            pickle.dump(dummy_model, f)
        with open('normalizer.pkl', 'wb') as f:
            pickle.dump(dummy_scaler, f)
        
        print("Dummy models created. For production, please train proper models.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)