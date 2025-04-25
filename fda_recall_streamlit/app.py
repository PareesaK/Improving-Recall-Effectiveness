import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from utils import clean_text, extract_recall_flags, categorize_distribution
from datetime import datetime
# Set page configuration
st.set_page_config(
    page_title="FDA Recall Classification Predictor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Add custom CSS
with open('fda_recall_streamlit/css/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Load the model and encoder
@st.cache_resource
def load_model():
    model = joblib.load('fda_recall_streamlit/models/random_forest_final_model.joblib')
    label_encoder = joblib.load('fda_recall_streamlit/models/label_encoder.joblib')
    return model, label_encoder
model, label_encoder = load_model()
# State to region mapping
state_region_map = {
    # Northeast
    'ME': 'Northeast', 'NH': 'Northeast', 'VT': 'Northeast', 'MA': 'Northeast',
    'RI': 'Northeast', 'CT': 'Northeast', 'NY': 'Northeast', 'NJ': 'Northeast', 'PA': 'Northeast',
    # Midwest
    'OH': 'Midwest', 'MI': 'Midwest', 'IN': 'Midwest', 'WI': 'Midwest', 'IL': 'Midwest',
    'MN': 'Midwest', 'IA': 'Midwest', 'MO': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    'NE': 'Midwest', 'KS': 'Midwest',
    # South
    'DE': 'South', 'MD': 'South', 'VA': 'South', 'WV': 'South', 'KY': 'South', 'NC': 'South',
    'SC': 'South', 'TN': 'South', 'GA': 'South', 'FL': 'South', 'AL': 'South', 'MS': 'South',
    'AR': 'South', 'LA': 'South', 'TX': 'South', 'OK': 'South', 'DC': 'South', 'PR': 'South',
    # West
    'MT': 'West', 'ID': 'West', 'WY': 'West', 'CO': 'West', 'NM': 'West', 'AZ': 'West',
    'UT': 'West', 'NV': 'West', 'CA': 'West', 'OR': 'West', 'WA': 'West', 'AK': 'West', 'HI': 'West',
    # Pacific Territories
    'GU': 'Territory'
}
# App title and description
st.title('FDA Recall Classification Predictor')
st.markdown("""
This application predicts the FDA recall classification (Class I, II, or III) based on recall details.
- **Class I:** Dangerous or defective products that could cause serious health problems or death
- **Class II:** Products that might cause a temporary health problem, or pose slight threat of a serious nature
- **Class III:** Products that are unlikely to cause any adverse health reaction, but violate FDA regulations
""")
# Create tabs for different app sections
tab1, tab2, tab3 = st.tabs(["Make Prediction", "Model Information", "Batch Processing"])
with tab1:
    st.header("Enter Recall Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_type = st.selectbox(
            "Product Type",
            ["Biologics", "Devices", "Drugs", "Food/Cosmetics", "Tobacco", "Veterinary"]
        )
        
        status = st.selectbox(
            "Status",
            ["Ongoing", "Terminated", "Completed"]
        )
        
        # Use empty placeholder for distribution pattern
        distribution_pattern = st.text_area(
            "Distribution Pattern",
            placeholder="Describe the distribution pattern (e.g., nationwide, international, regional)",
            height=100
        )
        
        recall_date = st.date_input(
            "Center Classification Date",
            datetime.now()
        )
        
    with col2:
        # Use empty placeholder for reason for recall
        reason_for_recall = st.text_area(
            "Reason for Recall",
            placeholder="Enter the detailed reason for the recall",
            height=150
        )
        
        recalling_country = st.selectbox(
            "Recalling Firm Country",
            ["United States", "International"]
        )
        
        # Show state selection only for US recalls
        if recalling_country == "United States":
            state_options = [
                'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 
                'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 
                'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 
                'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 
                'WA', 'WI', 'WV', 'WY', 'GU'
            ]
            recalling_firm_state = st.selectbox("Recalling Firm State", options=state_options)
        else:
            recalling_firm_state = ""
    
    # Process input when button is clicked
    if st.button("Predict Classification"):
        try:
            # Determine region based on state
            is_us = recalling_country == "United States"
            state = recalling_firm_state if is_us else ''
            region = state_region_map.get(state, 'International') if is_us else 'International'
            
            # Create a dataframe with the input
            input_data = pd.DataFrame({
                # Temporal features
                'Month_sin': [np.sin(2 * np.pi * recall_date.month/12)],
                'Month_cos': [np.cos(2 * np.pi * recall_date.month/12)],
                'DayOfWeek_sin': [np.sin(2 * np.pi * recall_date.weekday()/7)],
                'DayOfWeek_cos': [np.cos(2 * np.pi * recall_date.weekday()/7)],
                'Years_Since_First': [recall_date.year - 2019],  # Based on filtered data starting from 2019
                
                # Product Type features (one-hot encoded)
                'ProductType_Devices': [1 if product_type == "Devices" else 0],
                'ProductType_Drugs': [1 if product_type == "Drugs" else 0],
                'ProductType_Food/Cosmetics': [1 if product_type == "Food/Cosmetics" else 0],
                'ProductType_Tobacco': [1 if product_type == "Tobacco" else 0],
                'ProductType_Veterinary': [1 if product_type == "Veterinary" else 0],
                
                # Status features
                'Status_Ongoing': [1 if status == "Ongoing" else 0],
                'Status_Terminated': [1 if status == "Terminated" else 0],
                
                # Region features based on determined region
                'Region_International': [1 if region == 'International' else 0],
                'Region_Midwest': [1 if region == 'Midwest' else 0],
                'Region_Northeast': [1 if region == 'Northeast' else 0],
                'Region_South': [1 if region == 'South' else 0],
                'Region_West': [1 if region == 'West' else 0],
                
                # Distribution Scope
                'DistScope_Limited': [1 if "limited" in distribution_pattern.lower() else 0],
                'DistScope_Nationwide': [1 if "nationwide" in distribution_pattern.lower() else 0],
                'DistScope_Other': [1 if categorize_distribution(distribution_pattern) == "Other" else 0],
                'DistScope_Regional': [1 if "regional" in distribution_pattern.lower() else 0],
                
                # Text features
                'Reason_Word_Count': [len(reason_for_recall.split())]
            })
            
            # Extract recall flags
            recall_flags = extract_recall_flags(reason_for_recall)
            for flag, value in recall_flags.items():
                input_data[flag] = [value]
            
            # Fill any missing columns compared to training data
            expected_columns = [
                'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Years_Since_First', 
                'ProductType_Devices', 'ProductType_Drugs', 'ProductType_Food/Cosmetics', 
                'ProductType_Tobacco', 'ProductType_Veterinary', 'Status_Ongoing', 'Status_Terminated', 
                'Region_International', 'Region_Midwest', 'Region_Northeast', 'Region_South', 'Region_West', 
                'DistScope_Limited', 'DistScope_Nationwide', 'DistScope_Other', 'DistScope_Regional', 
                'Reason_Word_Count', 'has_listeria', 'has_salmonella', 'has_ecoli', 
                'has_undeclared', 'has_allergen', 'allergen_milk', 'allergen_soy', 'allergen_wheat', 
                'allergen_egg', 'allergen_peanut', 'allergen_nut', 'allergen_fish', 'allergen_shellfish', 
                'has_manufacturing_issue', 'has_quality_issue', 'has_mislabeling', 
                'has_foreign_material', 'possible_illness', 'possible_injury'
            ]
            
            for col in expected_columns:
                if col not in input_data.columns:
                    input_data[col] = 0
            
            # Ensure columns are in the right order
            input_data = input_data[expected_columns]
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Get the class name
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            class_probabilities = {label_encoder.inverse_transform([i])[0]: prob for i, prob in enumerate(prediction_proba)}
            
            # Display the prediction
            st.success(f"### Predicted Classification: {predicted_class}")
            
            # Display class probabilities
            st.subheader("Classification Probabilities")
            fig, ax = plt.subplots(figsize=(10, 5))
            classes = list(class_probabilities.keys())
            probs = list(class_probabilities.values())
            
            # Create color mapping based on class
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            bars = ax.bar(classes, probs, color=colors)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            
            # Add probability values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Show risk factors and key flags
            st.subheader("Identified Risk Factors")
            risk_factors = []
            
            for flag, value in recall_flags.items():
                if value == 1:
                    risk_factors.append(flag.replace('_', ' ').title())
            
            if risk_factors:
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.write("No specific risk factors identified in the recall reason.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
with tab2:
    st.header("Model Information")
    
    # Class-specific performance
    st.subheader("Class-Specific Performance")
    
    class_metrics = {
        'Class I': {'precision': 0.9343, 'recall': 0.9102, 'f1-score': 0.9221},
        'Class II': {'precision': 0.9444, 'recall': 0.9676, 'f1-score': 0.9559},
        'Class III': {'precision': 0.7611, 'recall': 0.6277, 'f1-score': 0.6880}
    }
    
    # Create a DataFrame for the class metrics
    class_df = pd.DataFrame.from_dict(class_metrics, orient='index')
    
    # Plot the class metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = class_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Class')
    ax.legend(title='Metric')
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    # Adjust y-axis to make room for labels
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
    
    st.pyplot(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Example feature importance from the model
    feature_importance = {
        'Reason_Word_Count': 0.140405,
        'Month_sin': 0.078714,
        'Month_cos': 0.073962,
        'Years_Since_First': 0.067180,
        'ProductType_Food/Cosmetics': 0.064854,
        'DayOfWeek_sin': 0.057964,
        'ProductType_Devices': 0.052548,
        'DayOfWeek_cos': 0.045465,
        'has_listeria': 0.041421,
        'DistScope_Other': 0.039157
    }
    
    # Create a DataFrame for feature importance
    feat_imp_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp_df.plot(kind='bar', ax=ax)
    ax.set_ylabel('Importance')
    ax.set_title('Top 10 Feature Importance')
    plt.tight_layout()
    st.pyplot(fig)
with tab3:
    st.header("Batch Processing")
    
    st.markdown("""
    Upload a CSV file with recall data to get predictions for multiple recalls at once.
    
    The CSV file should include the following columns:
    - Product Type (Biologics, Devices, Drugs, Food/Cosmetics, Tobacco, Veterinary)
    - Status (Ongoing, Terminated, Completed)
    - Distribution Pattern (text description)
    - Center Classification Date (MM/DD/YYYY format)
    - Reason for Recall (text description)
    - Recalling Firm Country (United States or other)
    - Recalling Firm State (Two-letter state code for US recalls, can be blank for international)
    """)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            input_df = pd.read_csv(uploaded_file)
            
            # Show the first few rows of the uploaded file
            st.subheader("Uploaded Data Preview")
            st.dataframe(input_df.head())
            
            # Process button
            if st.button("Process Batch"):
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize results DataFrame
                results_df = input_df.copy()
                results_df['PredictedClass'] = None
                results_df['ClassIProbability'] = None
                results_df['ClassIIProbability'] = None
                results_df['ClassIIIProbability'] = None
                
                # Process each row
                total_rows = len(input_df)
                
                for i, row in input_df.iterrows():
                    # Update progress
                    progress = int(100 * (i + 1) / total_rows)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing row {i+1}/{total_rows}")
                    
                    try:
                        # Create feature vector for this row
                        # Extract date components
                        # Convert MM/DD/YYYY format to datetime
                        recall_date = pd.to_datetime(row['Center Classification Date'], format='%m/%d/%Y', errors='coerce')
                        
                        # Determine region based on state
                        is_us = row['Recalling Firm Country'] == "United States"
                        state = str(row.get('Recalling Firm State', '')).upper() if is_us else ''
                        region = state_region_map.get(state, 'International') if is_us else 'International'
                        
                        # Create input features
                        features = {
                            # Temporal features
                            'Month_sin': np.sin(2 * np.pi * recall_date.month/12),
                            'Month_cos': np.cos(2 * np.pi * recall_date.month/12),
                            'DayOfWeek_sin': np.sin(2 * np.pi * recall_date.weekday()/7),
                            'DayOfWeek_cos': np.cos(2 * np.pi * recall_date.weekday()/7),
                            'Years_Since_First': recall_date.year - 2019,
                            
                            # Product Type features (one-hot encoded)
                            'ProductType_Devices': 1 if row['Product Type'] == "Devices" else 0,
                            'ProductType_Drugs': 1 if row['Product Type'] == "Drugs" else 0,
                            'ProductType_Food/Cosmetics': 1 if row['Product Type'] == "Food/Cosmetics" else 0,
                            'ProductType_Tobacco': 1 if row['Product Type'] == "Tobacco" else 0,
                            'ProductType_Veterinary': 1 if row['Product Type'] == "Veterinary" else 0,
                            
                            # Status features
                            'Status_Ongoing': 1 if row['Status'] == "Ongoing" else 0,
                            'Status_Terminated': 1 if row['Status'] == "Terminated" else 0,
                            
                            # Region features based on determined region
                            'Region_International': 1 if region == 'International' else 0,
                            'Region_Midwest': 1 if region == 'Midwest' else 0,
                            'Region_Northeast': 1 if region == 'Northeast' else 0,
                            'Region_South': 1 if region == 'South' else 0,
                            'Region_West': 1 if region == 'West' else 0,
                            
                            # Distribution Scope
                            'DistScope_Limited': 1 if "limited" in str(row['Distribution Pattern']).lower() else 0,
                            'DistScope_Nationwide': 1 if "nationwide" in str(row['Distribution Pattern']).lower() else 0,
                            'DistScope_Other': 1 if categorize_distribution(str(row['Distribution Pattern'])) == "Other" else 0,
                            'DistScope_Regional': 1 if "regional" in str(row['Distribution Pattern']).lower() else 0,
                            
                            # Text features
                            'Reason_Word_Count': len(str(row['Reason for Recall']).split())
                        }
                        
                        # Extract recall flags
                        recall_flags = extract_recall_flags(row['Reason for Recall'])
                        features.update(recall_flags)
                        
                        # Create a DataFrame with the features
                        input_data = pd.DataFrame([features])
                        
                        # Fill any missing columns compared to training data
                        expected_columns = [
                            'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Years_Since_First', 
                            'ProductType_Devices', 'ProductType_Drugs', 'ProductType_Food/Cosmetics', 
                            'ProductType_Tobacco', 'ProductType_Veterinary', 'Status_Ongoing', 'Status_Terminated', 
                            'Region_International', 'Region_Midwest', 'Region_Northeast', 'Region_South', 'Region_West', 
                            'DistScope_Limited', 'DistScope_Nationwide', 'DistScope_Other', 'DistScope_Regional', 
                            'Reason_Word_Count', 'has_listeria', 'has_salmonella', 'has_ecoli', 
                            'has_undeclared', 'has_allergen', 'allergen_milk', 'allergen_soy', 'allergen_wheat', 
                            'allergen_egg', 'allergen_peanut', 'allergen_nut', 'allergen_fish', 'allergen_shellfish', 
                            'has_manufacturing_issue', 'has_quality_issue', 'has_mislabeling', 
                            'has_foreign_material', 'possible_illness', 'possible_injury'
                        ]
                        
                        for col in expected_columns:
                            if col not in input_data.columns:
                                input_data[col] = 0
                        
                        # Ensure columns are in the right order
                        input_data = input_data[expected_columns]
                        
                        # Make prediction
                        prediction = model.predict(input_data)[0]
                        prediction_proba = model.predict_proba(input_data)[0]
                        
                        # Get the class name
                        predicted_class = label_encoder.inverse_transform([prediction])[0]
                        
                        # Store results
                        results_df.at[i, 'PredictedClass'] = predicted_class
                        results_df.at[i, 'ClassIProbability'] = prediction_proba[0]
                        results_df.at[i, 'ClassIIProbability'] = prediction_proba[1]
                        results_df.at[i, 'ClassIIIProbability'] = prediction_proba[2]
                        
                    except Exception as e:
                        # If there's an error processing this row, log it
                        results_df.at[i, 'PredictedClass'] = f"Error: {str(e)}"
                
                # Processing complete
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Display results
                st.subheader("Prediction Results")
                st.dataframe(results_df)
                
                # Prepare CSV for download
                csv = results_df.to_csv(index=False)
                
                # Add a download button for results
                st.download_button(
                    label="Download Prediction Results",
                    data=csv,
                    file_name="recall_predictions.csv",
                    mime="text/csv"
                )
                
                # Show summary statistics
                st.subheader("Summary Statistics")
                
                # Count predictions by class
                valid_predictions = results_df[results_df['PredictedClass'].str.contains('Error', na=True) == False]
                class_counts = valid_predictions['PredictedClass'].value_counts()
                
                # Create a bar chart instead of a pie chart
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Define colors for the classes
                colors = ['#ff9999', '#66b3ff', '#99ff99'][:len(class_counts)]
                
                # Create the bar chart
                bars = ax.bar(class_counts.index, class_counts.values, color=colors)
                
                # Add count labels on top of each bar
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height}', ha='center', va='bottom')
                
                # Add percentage labels inside each bar
                total = class_counts.sum()
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    percentage = (height / total) * 100
                    ax.text(bar.get_x() + bar.get_width()/2., height/2,
                            f'{percentage:.1f}%', ha='center', va='center',
                            color='black', fontweight='bold')
                
                # Customize the chart
                ax.set_xlabel('Classification')
                ax.set_ylabel('Number of Recalls')
                ax.set_title('Distribution of Predicted Classifications')
                
                # Ensure y-axis starts at 0
                ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # Add 10% padding for labels
                
                # Add a grid for better readability
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Show the chart
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error processing the uploaded file: {e}")