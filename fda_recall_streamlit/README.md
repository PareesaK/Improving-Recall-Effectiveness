# FDA Recall Classification Predictor

A Streamlit web application for predicting FDA recall classifications based on product and recall information.

## Features

- Single prediction interface for classifying individual recalls
- Batch processing for multiple recalls via CSV upload
- Visualization of model performance and feature importance
- Interactive user interface with real-time prediction

## Installation

1. Clone this repository
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Run the Streamlit app:
```
streamlit run app.py
```

## Data Requirements

For single predictions, enter the following information:
- Product Type (Biologics, Devices, Drugs, Food/Cosmetics, Tobacco, Veterinary)
- Status (Ongoing, Terminated, Completed)
- Distribution Pattern
- Recall Date
- Reason for Recall
- Recalling Country
- US Region (if applicable)

For batch processing, upload a CSV file with the following columns:
- ProductType
- Status
- DistributionPattern
- RecallDate (YYYY-MM-DD format)
- ReasonForRecall
- RecallingCountry
- USRegion

## Model Information

This application uses a Random Forest classifier trained on FDA recall data from 2019-2025. The model achieves:
- 93.18% accuracy
- 92.95% precision
- 93.18% recall
- 93.01% F1 score