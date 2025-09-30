# Disease Prediction from Medical Data - Task 4

**Objective:** Predict the possibility of diseases based on patient data using machine learning.

**Approach:** Apply classification techniques on structured medical datasets.

## Key Features
- Input features: Symptoms, age, blood test results, etc.
- Algorithms: SVM, Logistic Regression, Random Forest, XGBoost
- Datasets: Heart disease, Diabetes, Breast Cancer (from UCI ML Repository)

## Project Structure
- `src/` - Python scripts for training and prediction
- `data/` - Sample medical datasets (small version for testing)
- `requirements.txt` - Python dependencies
- `notebooks/` - Optional: EDA / model analysis notebooks
- `scripts/` - Optional: Download full datasets

## How to Run
1. Install dependencies:
pip install -r requirements.txt
2. Train the model:
python src/train.py --data data/sample_medical.csv --output models/model.pkl
3. Make predictions:
python src/predict.py --model models/model.pkl --input data/sample_medical.csv
Tested in **VS Code**.
