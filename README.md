# Improving-Recall-Effectiveness

#### -- Project Status: Active

## Data Source
The data used in this project was sourced from the [FDA Compliance Dashboards](https://datadashboard.fda.gov/ora/cd/recalls.htm), which contains information about products that have been recalled.

## 🗂️ Project Structure
```plaintext
Improving-Recall-Effectiveness/
├── code_library/
│   ├── Data Exploration.ipynb
│   ├── Data Preparation.ipynb
│   └── Modeling.ipynb
├── data/
│   └── recalls_details.xlsx
├── fda_recall_streamlit/
│   ├── models/
│   └── app.py
├── images/
│   ├── EDA_
│   ├── DATA_PREP_
│   └── MODEL_
└── other_materials/
    └── requirements.txt
```

## How to Set Up the Project Locally

### Step 1: Clone Repo

### Step 2: Navigate to the "Improving-Recall-Effectiveness" folder in command line

* activate your environment with conda

or 

* Create a virtual environment
```
python -m venv .venv
```

**To activate your virtual environment:**

> For Windows:
> ```
> .\.venv\Scripts\activate
> ```

> For macOS/Linux
> ```
> source .venv/bin/activate
> ```

### Step 3: Install dependencies
```
pip install -r other_materials/requirements.txt
```
# Overview: Early Detection of High-Risk Product Recalls  
*A Comparative Study of Multi-Class Classification Approaches*

## 📌 Project Intro / Objective
This project aims to enhance public health and regulatory efficiency by developing a machine learning system to classify the severity of FDA product recalls. Using structured metadata and unstructured recall descriptions, the system predicts whether a recall is Class I (most severe), Class II, or Class III. The ultimate goal is to enable early intervention and support faster, more accurate regulatory responses.

## 👥 Partner(s) / Contributor(s)
- `Lorena Dorado`  
- `Parisa Kamizi`

## 💻 Technologies
- **Python**
- **Streamlit**

## ⚙️ Methods Used
- **EDA**
- **Data Clean**: Removed ID columns and filled 1 missing value  
- **Data Preparation**:
  - Train-Test Split
  - Feature Engineering
  - Created pre-processed TRAINING dataset for modeling
- **Modeling**:
  - TRAINING dataset split for Cross Validation (create folds)
  - For each CV fold:
    - SMOTE on TRAINING portion
    - Feature selection on training portion
    - Train model with selected features
    - Hyperparameter tuning
    - Evaluate on validation portion of fold
- **Model Selection**:
  - Select best model and feature set based on CV results
  - Train best model on full training dataset using selected features
- **Model Evaluation on Test Set**
  - Cross-Validation and Learning Curve Analysis
- **Web App Deployment using Streamlit

## 📂 Project Description
This capstone project explores machine learning and NLP techniques to classify the risk level of FDA product recalls based on a combination of structured data (e.g., product type, distribution patterns) and unstructured text (e.g., recall reason). The study evaluates several classification algorithms including Logistic Regression, Decision Tree, Random Forest, XGBoost, and MLP. It also introduces a web-based dashboard to enable real-time risk classification, offering regulatory agencies and quality assurance professionals an actionable tool for recall management.

## 🧠 Datasets Used
- **Source:** U.S. Food and Drug Administration (FDA) recall database  
- **Size:** ~95,082 records  
- **Period:** 2020 onward  
- **Fields:** Product classification, status, recall reason, dates, manufacturer info, and more

## 🏆 Results
- **Best Model:** Random Forest  
- **F1 Score:** 0.9308 (overall weighted)  
- **Key Findings:**
- my-project/
├── README.md
├── images/
│   └── modeling.png
mod
- **Classification Performance (Test Set):**

| Class     | Precision | Recall  | F1 Score | Support |
|-----------|-----------|---------|----------|---------|
| Class I   | 0.9368    | 0.9144  | 0.9255   | 1,671   |
| Class II  | 0.9450    | 0.9671  | 0.9560   | 5,655   |
| Class III | 0.7566    | 0.6296  | 0.6873   | 548     |
| **Weighted Avg** | **0.9302** | **0.9324** | **0.9308** | **7,874** |
- **Deployment:** Interactive Streamlit App  
  - 🔗 [Live App](https://ads599-recall-classification.streamlit.app/)

## 📊 Key Features of the Web App
- Real-time single prediction interface
- Batch processing via CSV uploads
- Model explanation visuals (feature importance, performance metrics)

## 🧪 Models Compared
- **random_forest**
- **decision_tree**
- **mlp**
- **xgboost**
- **logreg**


## ⚠️ Limitations
- Slower training times due to model complexity
- Incomplete labeling on FDA website limited some analysis (e.g., root cause of recall initiation)

## 🔮 Future Work
- Improve Class III recall performance
- Incorporate root cause scraping and more NLP refinements
- Optimize for real-time scalability and low-latency deployment

## Acknowledgments
Code notebooks, documentation, and Streamlit application were improved with assistance from Claude (Anthropic), which helped with code optimization and analysis techniques.

## References
Anthropic. (2025). Claude 3.7 Sonnet (February 2025 version) [Large language model]. https://claude.ai

Barbosa-Slivinskis, V., Agi-Maluli, I., & Seth-Broder, J. (2024). A machine learning algorithm to predict medical device recall by the Food and Drug Administration. The Western Journal of Emergency Medicine, 26(1), 161–170. https://doi.org/10.5811/westjem.21238

Kuhn, M., & Johnson, K. (2013). Applied predictive modeling. Springer.
