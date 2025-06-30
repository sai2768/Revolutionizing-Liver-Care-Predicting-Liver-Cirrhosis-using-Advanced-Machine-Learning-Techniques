
# Liver Cirrhosis Prediction API

## Overview
This Flask-based API allows prediction of liver cirrhosis risk using 10 clinical laboratory features.
It uses a trained machine learning model (Random Forest Classifier) and is deployed as part of a liver care prediction web application.

---

## Base URL
```
http://127.0.0.1:5000/
```

---

## Endpoint: `/api/predict`
- **Method:** POST
- **Content-Type:** `application/json`
- **Description:** Returns the risk prediction for liver cirrhosis based on user input.

---

### ✅ Request Body (JSON)

```json
{
  "Age": 45,
  "Gender": 1,
  "Total_Bilirubin": 1.2,
  "Direct_Bilirubin": 0.4,
  "Alkaline_Phosphotase": 200,
  "Alamine_Aminotransferase": 35,
  "Aspartate_Aminotransferase": 50,
  "Total_Protiens": 6.5,
  "Albumin": 3.1,
  "Albumin_and_Globulin_Ratio": 1.0
}
```

---

### 🔁 Response Body (JSON)
```json
{
  "success": true,
  "prediction": 1,
  "confidence": 87.56,
  "probabilities": {
    "no_cirrhosis": 12.44,
    "cirrhosis": 87.56
  }
}
```

- `prediction`: 0 = Low Risk, 1 = High Risk
- `confidence`: Highest probability score of the predicted class

---

### ❌ Error Responses
- `400 Bad Request`: Input is missing or invalid
- `500 Internal Server Error`: Model or server-side issue

---

## Technology Stack
- Python 3.x
- Flask (Backend Framework)
- Scikit-learn (ML Model)
- Pickle (Model Serialization)
- HTML/CSS/Jinja2 (Frontend Templates)

---

## Author
Developed by [Your Name]  
Intern at SmartInternz Platform  
Project Title: *"Revolutionizing Liver Care: Predicting Liver Cirrhosis using Advanced Machine Learning Techniques"*
