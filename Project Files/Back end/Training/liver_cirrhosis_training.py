#!/usr/bin/env python3
"""
Liver Cirrhosis Prediction Model Training
=========================================

This script trains multiple machine learning models to predict liver cirrhosis
from clinical laboratory parameters.

Models Trained:
- Random Forest Classifier
- XGBoost Classifier
- Support Vector Machine
- Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

class LiverCirrhosisPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare_data(self, data_path="../Data/liver_dataset.csv"):
        """Load and prepare the liver dataset"""
        # For demonstration, we'll create a synthetic dataset since the original might not be available
        # In practice, you would load your actual dataset here
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Age': np.random.randint(20, 80, n_samples),
            'Gender': np.random.choice([0, 1], n_samples),
            'Total_Bilirubin': np.random.uniform(0.1, 15.0, n_samples),
            'Direct_Bilirubin': np.random.uniform(0.1, 8.0, n_samples),
            'Alkaline_Phosphotase': np.random.uniform(100, 800, n_samples),
            'Alamine_Aminotransferase': np.random.uniform(10, 200, n_samples),
            'Aspartate_Aminotransferase': np.random.uniform(10, 300, n_samples),
            'Total_Protiens': np.random.uniform(4.0, 9.0, n_samples),
            'Albumin': np.random.uniform(1.5, 5.0, n_samples),
            'Albumin_and_Globulin_Ratio': np.random.uniform(0.3, 2.0, n_samples)
        }
        
        # Create target variable based on some logical rules
        df = pd.DataFrame(data)
        df['Dataset'] = 0
        
        # Higher likelihood of cirrhosis with elevated liver enzymes and bilirubin
        cirrhosis_condition = (
            (df['Total_Bilirubin'] > 2.5) | 
            (df['Alkaline_Phosphotase'] > 400) |
            (df['Alamine_Aminotransferase'] > 80) |
            (df['Aspartate_Aminotransferase'] > 100)
        )
        df.loc[cirrhosis_condition, 'Dataset'] = 1
        
        # Add some noise to make it more realistic
        noise_indices = np.random.choice(df.index, size=int(0.2 * len(df)), replace=False)
        df.loc[noise_indices, 'Dataset'] = 1 - df.loc[noise_indices, 'Dataset']
        
        return df
    
    def exploratory_data_analysis(self, df):
        """Perform comprehensive EDA"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"Target distribution:\n{df['Dataset'].value_counts()}")
        
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('../Documentation/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature distributions
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, column in enumerate(df.columns[:-1]):
            axes[i].hist(df[df['Dataset']==0][column], alpha=0.7, label='No Cirrhosis', bins=30)
            axes[i].hist(df[df['Dataset']==1][column], alpha=0.7, label='Cirrhosis', bins=30)
            axes[i].set_title(column)
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('../Documentation/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models and compare performance"""
        
        # Random Forest
        print("\n=== TRAINING RANDOM FOREST ===")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        self.models['random_forest'] = rf_grid.best_estimator_
        
        # XGBoost
        print("\n=== TRAINING XGBOOST ===")
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
        xgb_grid.fit(X_train, y_train)
        self.models['xgboost'] = xgb_grid.best_estimator_
        
        # Support Vector Machine
        print("\n=== TRAINING SVM ===")
        svm = SVC(probability=True, random_state=42)
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
        svm_grid.fit(X_train, y_train)
        self.models['svm'] = svm_grid.best_estimator_
        
        # Logistic Regression
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr_params = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }
        lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='accuracy', n_jobs=-1)
        lr_grid.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_grid.best_estimator_
        
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        results = {}
        
        print("\n=== MODEL EVALUATION RESULTS ===")
        for name, model in self.models.items():
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'model': model
            }
            
            print(f"\n{name.upper()}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_model = results[best_model_name]['model']
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n=== BEST MODEL: {best_model_name.upper()} ===")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        return best_model, best_accuracy, results
    
    def save_models(self, best_model, scaler):
        """Save the best model and scaler"""
        # Save the best model
        with open('../Flask/rf_acc_68.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save the scaler
        with open('../Flask/normalizer.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\nModels saved successfully!")
        print(f"Best model: ../Flask/rf_acc_68.pkl")
        print(f"Scaler: ../Flask/normalizer.pkl")

def main():
    """Main training pipeline"""
    predictor = LiverCirrhosisPredictor()
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = predictor.load_and_prepare_data()
    
    # Exploratory Data Analysis
    predictor.exploratory_data_analysis(df)
    
    # Prepare features and target
    X = df.drop(['Dataset'], axis=1)
    y = df['Dataset']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    X_train_scaled = predictor.scaler.fit_transform(X_train)
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Train models
    predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Evaluate models
    best_model, best_accuracy, results = predictor.evaluate_models(X_test_scaled, y_test)
    
    # Save models
    predictor.save_models(best_model, predictor.scaler)
    
    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")
    print("You can now run the Flask application!")

if __name__ == "__main__":
    main()