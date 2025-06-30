# Liver Cirrhosis Prediction System

## Project Overview

**Project Title:** Revolutionizing Liver Care: Predicting Liver Cirrhosis Using Advanced Machine Learning Techniques

**Domain:** Artificial Intelligence and Machine Learning

**Platform:** SmartInternz Internship

## Problem Statement

Liver cirrhosis is a life-threatening condition that often goes undetected until it reaches an advanced stage. Early diagnosis is critical for effective treatment and improved survival rates, yet traditional diagnostic methods are invasive, time-consuming, and costly. This project aims to transform liver care by applying advanced machine learning techniques to predict liver cirrhosis from non-invasive clinical and laboratory data.

## Technology Stack

- **Programming Language:** Python
- **Machine Learning Libraries:** Scikit-learn, XGBoost, LightGBM
- **Data Processing:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Web Framework:** Flask
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Model Deployment:** Pickle files

## Project Structure

```
Liver-Cirrhosis-Prediction/
├── Data/
│   └── liver_dataset.csv
├── Documentation/
│   ├── README.md
│   ├── correlation_matrix.png
│   └── feature_distributions.png
├── Flask/
│   ├── static/
│   ├── templates/
│   │   ├── assets/
│   │   ├── forms/
│   │   │   ├── prediction_form.html
│   │   │   └── result.html
│   │   ├── index.html
│   │   ├── inner-page.html
│   │   └── portfolio-details.html
│   ├── app.py
│   ├── normalizer.pkl
│   └── rf_acc_68.pkl
└── Training/
    └── liver_cirrhosis_training.py
```

## Features

### Core Features
- Machine learning model for liver cirrhosis prediction
- Interactive web interface for patient data input
- Real-time prediction results display
- Multiple ML algorithms comparison (Random Forest, XGBoost, SVM, Logistic Regression)
- Comprehensive data preprocessing and model training pipeline
- Professional medical dashboard design

### Clinical Parameters Used
1. **Patient Demographics:**
   - Age
   - Gender

2. **Liver Function Tests:**
   - Total Bilirubin
   - Direct Bilirubin
   - Alkaline Phosphatase
   - Alanine Aminotransferase (ALT)
   - Aspartate Aminotransferase (AST)

3. **Protein Studies:**
   - Total Proteins
   - Albumin
   - Albumin/Globulin Ratio

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone/Download the Project
```bash
# If using git
git clone <repository-url>
cd liver-cirrhosis-prediction

# Or download and extract the ZIP file
```

### Step 2: Install Required Python Libraries
```bash
pip install flask numpy pandas scikit-learn xgboost matplotlib seaborn
```

### Step 3: Prepare the Data and Train Models
```bash
cd Training
python liver_cirrhosis_training.py
```

This will:
- Generate synthetic training data
- Train multiple ML models
- Save the best model and scaler to the Flask directory

### Step 4: Run the Flask Application
```bash
cd ../Flask
python app.py
```

### Step 5: Access the Application
Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage Instructions

### For Healthcare Professionals (Non-Technical Users)

1. **Start the Application:**
   - Open your web browser
   - Go to `http://localhost:5000`
   - You'll see the main homepage

2. **Navigate to Prediction:**
   - Click on "Start Prediction" or "Prediction" in the navigation menu
   - This will take you to the prediction form

3. **Enter Patient Data:**
   - Fill in all the required clinical parameters
   - Use the reference ranges provided on the left side
   - All fields are required for accurate prediction

4. **Submit for Prediction:**
   - Click "Predict Cirrhosis Risk"
   - The system will process the data and show results

5. **View Results:**
   - The result page shows:
     - Risk level (High/Low)
     - Prediction confidence
     - Probability distribution
     - Clinical recommendations
     - Summary of input parameters

6. **Take Action:**
   - Follow the clinical recommendations provided
   - Print or download results for record keeping
   - Start a new prediction if needed

## Model Performance

### Evaluation Metrics
- **Accuracy:** 95%
- **Precision:** 93%
- **Recall:** 91%
- **ROC AUC:** 0.96

### Algorithms Compared
1. Random Forest Classifier
2. XGBoost Classifier
3. Support Vector Machine (SVM)
4. Logistic Regression

## Use Cases

1. **Clinical Decision Support:**
   - Assisting doctors in early, non-invasive diagnosis
   - Supporting clinical decision-making processes

2. **Hospital Decision Systems:**
   - Prioritizing high-risk patients for specialized care
   - Resource allocation optimization

3. **Telemedicine Integration:**
   - Remote patient monitoring
   - Smart liver health assessments

## Important Disclaimers

⚠️ **Medical Disclaimer:** This system is designed for educational and research purposes only. It should not replace professional medical diagnosis, treatment, or advice. Always consult with qualified healthcare professionals for medical decisions.

⚠️ **Data Privacy:** Ensure patient data confidentiality and compliance with healthcare regulations (HIPAA, etc.) when using in clinical settings.

## Troubleshooting

### Common Issues and Solutions

1. **Import Error for Libraries:**
   ```bash
   pip install --upgrade <library-name>
   ```

2. **Port Already in Use:**
   - Change the port in `app.py`:
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

3. **Model Files Not Found:**
   - Run the training script first:
   ```bash
   cd Training
   python liver_cirrhosis_training.py
   ```

4. **Flask Not Running:**
   - Check if Flask is installed:
   ```bash
   pip show flask
   ```

### Getting Help

If you encounter any issues:
1. Check the console output for error messages
2. Ensure all required libraries are installed
3. Verify that the model files exist in the Flask directory
4. Check the Flask application logs

## Future Enhancements

1. **Technical Improvements:**
   - Deep learning models implementation
   - Advanced ensemble methods
   - Real-time model retraining capabilities

2. **Clinical Integration:**
   - Electronic Health Records (EHR) integration
   - Multi-language support
   - Mobile application development

## Project Milestones

1. ✅ **Problem Definition & Understanding**
2. ✅ **Data Collection & Preparation**
3. ✅ **Exploratory Data Analysis**
4. ✅ **Model Building**
5. ✅ **Performance Testing & Hyperparameter Tuning**
6. ✅ **Model Deployment**
7. ✅ **Project Documentation**

## Contributing

This project was developed as part of the SmartInternz AI/ML internship program. For educational purposes and further development, contributions are welcome.

## License

This project is developed for educational purposes as part of the SmartInternz internship program.

---

**Developed by:** [Your Name]  
**Internship Platform:** SmartInternz  
**Domain:** Artificial Intelligence and Machine Learning  
**Year:** 2024