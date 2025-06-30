# Detailed Setup Instructions for Liver Cirrhosis Prediction System

## For Non-Technical Users (Step-by-Step Guide)

### Overview
This guide will help you set up and run the Liver Cirrhosis Prediction System on your computer. The system uses artificial intelligence to predict liver cirrhosis risk based on blood test results.

### What You Need
- A computer with Windows, Mac, or Linux
- Internet connection for downloading required software
- About 30 minutes for complete setup

## Step 1: Install Python

### For Windows:
1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.11" (or latest version)
3. Run the downloaded file
4. **IMPORTANT:** Check "Add Python to PATH" during installation
5. Click "Install Now"
6. Wait for installation to complete

### For Mac:
1. Go to https://www.python.org/downloads/
2. Download Python for macOS
3. Open the downloaded .pkg file
4. Follow the installation wizard

### For Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Verify Python Installation:
1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
2. Type: `python --version` or `python3 --version`
3. You should see something like "Python 3.11.0"

## Step 2: Download the Project

### Option A: Download ZIP file
1. Download the project ZIP file
2. Extract it to a folder (e.g., Desktop/liver-prediction)
3. Remember the folder location

### Option B: Using Git (if available)
```bash
git clone <repository-url>
cd liver-cirrhosis-prediction
```

## Step 3: Install Required Software Libraries

1. Open Command Prompt/Terminal
2. Navigate to your project folder:
   ```bash
   cd path/to/your/project/folder
   ```
3. Install required libraries:
   ```bash
   pip install flask numpy pandas scikit-learn xgboost matplotlib seaborn
   ```
4. Wait for installation to complete (may take 5-10 minutes)

## Step 4: Prepare the Machine Learning Models

1. In Command Prompt/Terminal, go to the Training folder:
   ```bash
   cd Training
   ```
2. Run the training script:
   ```bash
   python liver_cirrhosis_training.py
   ```
3. Wait for training to complete (2-5 minutes)
4. You should see messages about model training and saving

## Step 5: Start the Web Application

1. Go to the Flask folder:
   ```bash
   cd ../Flask
   ```
2. Start the application:
   ```bash
   python app.py
   ```
3. You should see messages like:
   ```
   * Running on http://127.0.0.1:5000
   * Debug mode: on
   ```

## Step 6: Use the Application

1. Open your web browser (Chrome, Firefox, Safari, etc.)
2. Go to: `http://localhost:5000`
3. You should see the Liver Care AI homepage

## Using the Prediction System

### Step 1: Navigate to Prediction
- Click "Start Prediction" or "Prediction" in the menu

### Step 2: Enter Patient Information
Fill in all the required fields:

**Patient Demographics:**
- Age: Enter patient age in years
- Gender: Select Male or Female

**Blood Test Results (get these from lab reports):**
- Total Bilirubin: Normal range 0.2-1.2 mg/dL
- Direct Bilirubin: Normal range 0.0-0.3 mg/dL
- Alkaline Phosphatase: Normal range 44-147 U/L
- ALT: Normal range 7-56 U/L
- AST: Normal range 10-40 U/L
- Total Proteins: Normal range 6.0-8.3 g/dL
- Albumin: Normal range 3.5-5.0 g/dL
- A/G Ratio: Normal range 1.1-2.5

### Step 3: Get Prediction
- Click "Predict Cirrhosis Risk"
- Wait for results (usually instant)

### Step 4: Understand Results
The system will show:
- **Risk Level:** High or Low
- **Confidence:** How sure the AI is about the prediction
- **Probabilities:** Percentage chances for each outcome
- **Recommendation:** What to do next

## Troubleshooting Common Issues

### Issue 1: "Python is not recognized"
**Solution:**
- Reinstall Python and make sure to check "Add Python to PATH"
- Restart Command Prompt/Terminal

### Issue 2: "pip is not recognized"
**Solution:**
- Try using `python -m pip` instead of just `pip`
- Or reinstall Python with pip included

### Issue 3: Library installation fails
**Solution:**
- Try installing one by one:
  ```bash
  pip install flask
  pip install numpy
  pip install pandas
  pip install scikit-learn
  pip install xgboost
  pip install matplotlib
  pip install seaborn
  ```

### Issue 4: "Address already in use"
**Solution:**
- The application might already be running
- Close other instances or change the port in app.py

### Issue 5: Model files not found
**Solution:**
- Make sure you ran the training script:
  ```bash
  cd Training
  python liver_cirrhosis_training.py
  ```

### Issue 6: Web page won't load
**Solution:**
- Make sure the Flask app is running
- Try `http://127.0.0.1:5000` instead of `localhost:5000`
- Check firewall settings

## Important Notes

### For Healthcare Use:
- This tool is for educational purposes only
- Always consult with medical professionals
- Do not use for actual medical diagnosis
- Verify all lab values before entering

### Data Security:
- Patient data is not stored by the system
- Use appropriate data protection measures
- Follow your organization's privacy policies

### System Requirements:
- **RAM:** Minimum 4GB, recommended 8GB
- **Storage:** At least 2GB free space
- **Internet:** Required for initial setup only

## Getting Help

If you're still having trouble:

1. **Check Error Messages:** Copy the exact error message
2. **Verify Steps:** Make sure you followed each step
3. **Contact Support:** Provide details about:
   - Your operating system
   - Python version
   - Exact error messages
   - Which step failed

## Quick Reference Commands

```bash
# Check Python version
python --version

# Install libraries
pip install flask numpy pandas scikit-learn xgboost matplotlib seaborn

# Train models
cd Training
python liver_cirrhosis_training.py

# Run application
cd ../Flask
python app.py
```

## Web Application URLs

- **Homepage:** http://localhost:5000
- **Prediction Form:** http://localhost:5000/prediction
- **About Page:** http://localhost:5000/about
- **Portfolio:** http://localhost:5000/portfolio

---

**Support Note:** This system was developed for the SmartInternz AI/ML internship program. For technical support, please refer to your internship coordinator or the provided documentation.