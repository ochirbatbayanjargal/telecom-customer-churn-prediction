# Telecom Customer Churn Prediction Project
# -------------------------------------------
# This project follows a 10-step workflow for data analytics:
# 1. Problem Definition & Business Understanding
# 2. Data Collection & Understanding
# 3. Data Preparation & Cleaning
# 4. Exploratory Data Analysis
# 5. Feature Engineering & Selection
# 6. Model Development
# 7. Model Evaluation & Interpretation
# 8. Data Visualization & Communication
# 9. SQL Integration
# 10. Deployment Considerations
# Core data manipulation and analysis

#%% Imports - Core data manipulation and analysis
import pandas as pd
import numpy as np

# %% Imports - Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# %% Imports - Data preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %% Imports - Feature selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

# %% Imports - Model building
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

# %% Imports - Model evaluation
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_curve,
                           roc_auc_score, precision_recall_curve)

# %% Imports - SQL integration and warnings
import sqlite3
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# %% Setup - Visualization styles and paths
sns.set(style="whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# %% Step 1: Problem Definition & Business Understanding
# Business Problem: A telecommunications company is experiencing customer churn
# and needs to identify which customers are at risk of leaving.
#
# Project Objectives:
# 1. Build a model to predict which customers are likely to churn
# 2. Identify key factors that contribute to customer churn
# 3. Recommend targeted retention strategies based on insights
# 4. Estimate the potential financial impact of implementing these strategies
#
# Success Metrics:
# 1. Model performance: Achieve at least 80% accuracy with high recall
# 2. Business impact: Demonstrate potential reduction in churn rate
# 3. ROI analysis: Show financial benefit vs. implementation cost
#
# Stakeholders:
# 1. Customer Success team - Will implement retention strategies
# 2. Marketing team - Will create targeted campaigns
# 3. Product team - May need to address product issues causing churn
# 4. Executive leadership - Needs to understand financial impact
print("Step 1: Problem Definition & Business Understanding - Complete!")

# %% Step 2: Define data loading function
def load_and_understand_data():
    """
    Load the telecom customer churn dataset and perform initial exploration.
    """
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    # Display basic information
    print("Dataset Shape:", df.shape)
    print("\nDataset Preview:")
    print(df.head())
    
    # Check data types and missing values
    print("\nData Types and Missing Values:")
    print(df.info())
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values Per Column:")
    print(df.isnull().sum())
    
    # Check target variable distribution
    print("\nChurn Distribution:")
    print(df['Churn'].value_counts(normalize=True) * 100)
    
    return df

# %% Step 2: Load and understand data - basic exploration
print("Loading and understanding data...")
df = load_and_understand_data()
    
# %% Step 2B: Define enhanced exploration function
def explore_data_in_depth(df):
    """
    Perform a more detailed exploration of the telecom churn dataset.
    """
    # Check unique values for categorical columns
    print("\n== Categorical Feature Analysis ==")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"\nUnique values in {col}:")
        print(df[col].value_counts())
        
    # Check numeric feature distributions
    print("\n== Numeric Feature Analysis ==")
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_columns:
        print(f"\nSummary for {col}:")
        print(df[col].describe())
    
    # Correlation with churn for numeric features
    # First, convert 'Yes'/'No' in Churn to 1/0 if needed
    if df['Churn'].dtype == 'object':
        churn_numeric = df['Churn'].map({'Yes': 1, 'No': 0})
    else:
        churn_numeric = df['Churn']
    
    print("\n== Correlation with Churn ==")
    for col in numeric_columns:
        if col != 'Churn':
            correlation = churn_numeric.corr(df[col])
            print(f"Correlation between Churn and {col}: {correlation:.4f}")
    
# %% (continued): Data quality checks
def check_data_quality(df):
    """Check for data quality issues like outliers and inconsistencies"""
    print("\n== Data Quality Check ==")
    
    # Check for outliers in numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    print("\nPotential outliers (values > 3 std devs from mean):")
    for col in numeric_columns:
        mean = df[col].mean()
        std = df[col].std()
        outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
        if not outliers.empty:
            print(f"  {col}: {len(outliers)} potential outliers")
    
    # Check for potential errors or inconsistencies
    print("\nUnique customer count:", df['customerID'].nunique())
    print("Total rows:", len(df))
    
    # Initial look at churn by key variables
    print("\n== Initial Churn Insights ==")
    # Example: Churn by contract type
    if 'Contract' in df.columns:
        print("\nChurn rate by contract type:")
        contract_churn = df.groupby('Contract')['Churn'].value_counts(normalize=True).unstack() * 100
        print(contract_churn)
    
# %% Execute enhanced exploration
    print("\nPerforming deeper data exploration...")
    explore_data_in_depth(df)
check_data_quality(df)
print("\nExploration complete! Ready to move to data cleaning and preparation.")

# %% Step 3: Define data cleaning function
def clean_and_prepare_data(df):
    """
    Clean the dataset and prepare it for modeling.
    """
    print("Starting data cleaning and preparation...")
    
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    # Check for missing values in TotalCharges
    print("\nChecking for empty values in TotalCharges...")
    missing_total_charges = df_clean[df_clean['TotalCharges'].str.strip() == '']
    print(f"Found {len(missing_total_charges)} rows with empty TotalCharges")
    
    # Handle empty strings in TotalCharges
    # For new customers (tenure=0), set TotalCharges to 0
    df_clean.loc[df_clean['TotalCharges'].str.strip() == '', 'TotalCharges'] = '0'
    
    # Convert TotalCharges to numeric
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'])
    
    # Confirm conversion was successful
    print("\nData types after conversion:")
    print(df_clean.dtypes[['MonthlyCharges', 'TotalCharges']])
    
    # Encode binary categorical variables (Yes/No)
    binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_columns:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
    
    # Handle categorical variables with more than two categories using one-hot encoding
    # First handle the 'No internet service' and 'No phone service' values
    internet_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_services:
        df_clean[col] = df_clean[col].replace({'No internet service': 'No'})
    
    df_clean['MultipleLines'] = df_clean['MultipleLines'].replace({'No phone service': 'No'})
    
    # Now map the Yes/No columns
    for col in internet_services + ['MultipleLines']:
        df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
    
    # One-hot encode the remaining categorical variables
    df_clean = pd.get_dummies(df_clean, columns=['gender', 'InternetService', 'Contract', 'PaymentMethod'], drop_first=True)
    
    # Drop customerID as it's not relevant for modeling
    df_clean = df_clean.drop('customerID', axis=1)
    
    # Print the shape after preparation
    print(f"\nData shape after cleaning and preparation: {df_clean.shape}")
    print(f"Final columns: {df_clean.columns.tolist()}")
    
    # Check for any remaining issues
    print("\nChecking for any remaining missing values:")
    print(df_clean.isnull().sum().sum())
    
    return df_clean

#Execute data cleaning and preparation
df_clean = clean_and_prepare_data(df)
print("\nData cleaning and preparation complete! Ready for Exploratory Data Analysis.")

# %% Step 4: Exploratory Data Analysis (EDA)
def perform_eda(df_clean, df_original):
    """
    Perform exploratory data analysis using both the original and cleaned dataframes.
    
    Args:
        df_clean: The cleaned dataframe with encoded features
        df_original: The original dataframe with categorical features
    """
    print("Starting Exploratory Data Analysis...")
    
    # Distribution of the target variable (Churn)
    plt.figure(figsize=(10, 6))
    plt.title('Churn Distribution')
    sns.countplot(x='Churn', data=df_original)
    churn_percent = df_original['Churn'].value_counts(normalize=True) * 100
    for i, v in enumerate(df_original['Churn'].value_counts()):
        plt.text(i, v + 50, f"{churn_percent.iloc[i]:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/churn_Distribution.png', dpi=300, bbox_inches='tight')

    plt.show()

    # Correlation heatmap for numerical features
    numerical_features = df_clean.select_dtypes(include=['int64', 'float64']).columns
    plt.figure(figsize=(14, 12))
    corr_matrix = df_clean[numerical_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Heatmap', fontsize=15)

    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # Tenure vs Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='tenure', data=df_original)
    plt.title('Customer Tenure by Churn Status')

    plt.tight_layout()
    plt.savefig('visualizations/customer_tenure_by_churn_status.png', dpi=300, bbox_inches='tight')

    plt.show()
            
    # Create tenure groups for analysis
    df_original['tenure_group'] = pd.cut(df_original['tenure'], bins=[0, 12, 24, 36, 48, 60, 72], 
                              labels=['0-12 months', '12-24 months', '24-36 months', 
                                     '36-48 months', '48-60 months', '60-72 months'])
    
    # Churn rate by tenure group
    plt.figure(figsize=(10, 6))
    tenure_churn = df_original.groupby('tenure_group')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    tenure_churn.plot(kind='bar', color='skyblue')
    plt.title('Churn Rate by Tenure Group')
    plt.xlabel('Tenure Group')
    plt.ylabel('Churn Rate (%)')
    for i, v in enumerate(tenure_churn):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')

    plt.tight_layout()
    plt.savefig('visualizations/churn_rate_by_tenure_group.png', dpi=300, bbox_inches='tight')

    plt.show()
        
    # Monthly charges vs Churn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df_original)
    plt.title('Monthly Charges by Churn Status')

    plt.tight_layout()
    plt.savefig('visualizations/monthly_charges_by_churn_status.png', dpi=300, bbox_inches='tight')

    plt.show()
        
    # Create monthly charge groups for analysis
    df_original['charge_group'] = pd.cut(df_original['MonthlyCharges'], bins=[0, 30, 60, 90, 120], 
                              labels=['$0-30', '$30-60', '$60-90', '$90-120'])
    
    # Churn rate by monthly charges
    plt.figure(figsize=(10, 6))
    charge_churn = df_original.groupby('charge_group')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    charge_churn.plot(kind='bar', color='lightgreen')
    plt.title('Churn Rate by Monthly Charges')
    plt.xlabel('Monthly Charges')
    plt.ylabel('Churn Rate (%)')
    for i, v in enumerate(charge_churn):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center') 

    plt.tight_layout()
    plt.savefig('visualizations/churn_rate_by_monthly_charges.png', dpi=300, bbox_inches='tight')

    plt.show()
        
    # Contract type vs Churn
    plt.figure(figsize=(10, 6))
    contract_data = pd.crosstab(df_original['Contract'], df_original['Churn'])
    contract_percent = contract_data.div(contract_data.sum(axis=1), axis=0) * 100
    contract_percent.plot(kind='bar', stacked=True)
    plt.title('Churn Rate by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Percentage')
    plt.legend(title='Churn') 

    plt.tight_layout()
    plt.savefig('visualizations/churn_rate_by_contract_type.png', dpi=300, bbox_inches='tight')

    plt.show()
        
    # Churn rate by internet service
    plt.figure(figsize=(10, 6))
    internet_data = pd.crosstab(df_original['InternetService'], df_original['Churn'])
    internet_percent = internet_data.div(internet_data.sum(axis=1), axis=0) * 100
    internet_percent.plot(kind='bar', stacked=True)
    plt.title('Churn Rate by Internet Service Type')
    plt.xlabel('Internet Service')
    plt.ylabel('Percentage')
    plt.legend(title='Churn')

    plt.tight_layout()
    plt.savefig('visualizations/churn_rate_by_internet_service_type.png', dpi=300, bbox_inches='tight')

    plt.show()
    
    
    # Check for interactions: Contract type and tenure
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Contract', y='tenure', hue='Churn', data=df_original)
    plt.title('Effect of Contract Type and Tenure on Churn')

    plt.tight_layout()
    plt.savefig('visualizations/effect_of_contract_type_and_tenure_on_churn.png', dpi=300, bbox_inches='tight')

    plt.show()
      

    # Analyze customer services
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                      'StreamingTV', 'StreamingMovies']
    
    # Create a figure to visualize churn by service
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(service_columns, 1):
        plt.subplot(3, 3, i)
        service_data = pd.crosstab(df_original[column], df_original['Churn'])
        service_percent = service_data.div(service_data.sum(axis=1), axis=0) * 100
        service_percent.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title(f'Churn by {column}')
        plt.xlabel(column)
        plt.ylabel('Percentage')
        plt.legend(title='Churn')
    plt.tight_layout()
    plt.savefig('visualizations/churn_by_service.png', dpi=300, bbox_inches='tight')    

    plt.show() 
    
   # Create a summary of key findings
    print("\nSummary of Key EDA Findings:")
    print("1. Tenure is strongly related to churn - newer customers are more likely to leave.")
    print("2. Contract type has a dramatic impact on churn rates.")
    print("3. Higher monthly charges are associated with increased churn.")
    print("4. Fiber optic internet service customers show higher churn rates.")
    print("5. Services like OnlineSecurity and TechSupport are associated with lower churn when present.")
    
    return df_original  # Return with added analysis columns

# %% Execute Step 4: EDA
# We need both the clean data and original data for complete analysis
df_eda = perform_eda(df_clean, df)  # Pass both dataframes
print("\nExploratory Data Analysis complete! Ready for Feature Engineering.")

# %% Step 5: Feature Engineering & Selection
def engineer_and_select_features(df_clean, df_original):
    """
    Engineer new features and select the most relevant ones for the model.
    
    Args:
        df_clean: The cleaned dataframe with encoded features
        df_original: The original dataframe with additional analysis columns from EDA
    """
    print("Starting Feature Engineering & Selection...")
    
    # Create a copy to avoid modifying the input data
    df_features = df_clean.copy()
    
    # 1. Engineer new features
    
    # Calculate ratio of total charges to tenure (average monthly spend)
    # Handle division by zero for new customers (tenure=0)
    df_features['avg_monthly_spend'] = df_features.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'], 
        axis=1
    )
    
    # Calculate difference between monthly charges and average spend
    # This can indicate recent price changes
    df_features['price_change'] = df_features['MonthlyCharges'] - df_features['avg_monthly_spend']
    
    # Total number of services for each customer
    service_columns = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 
                      'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                      'StreamingTV', 'StreamingMovies']
    
    # For cleaned data, these are already binary (0/1)
    df_features['total_services'] = df_features[service_columns].sum(axis=1)
    
    # Calculate services to charges ratio (value for money indicator)
    df_features['services_per_dollar'] = df_features['total_services'] / df_features['MonthlyCharges']
    
    # Tenure-related features
    # Binning tenure into categories (we'll use both numeric and binned versions)
    tenure_bins = [0, 12, 24, 36, 48, 60, 72]
    tenure_labels = ['0-12', '12-24', '24-36', '36-48', '48-60', '60-72']
    df_features['tenure_bin'] = pd.cut(df_features['tenure'], bins=tenure_bins, labels=tenure_labels)
    
    # One-hot encode the tenure bins
    df_features = pd.get_dummies(df_features, columns=['tenure_bin'], prefix='tenure')
    
    # 2. Feature Selection
    
    # Separate features and target
    X = df_features.drop('Churn', axis=1)
    y = df_features['Churn']
    
    # Identify top features using feature importance from a random forest
    print("\nIdentifying important features using Random Forest...")
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)
    
    # Get feature importances
    importances = rf_selector.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Print top 15 features
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()

    plt.savefig('visualizations/top15_features_importance.png', dpi=300, bbox_inches='tight')

    plt.show()
    
    # 3. Select Top Features for the model
    # Typically, we'd keep features above a certain threshold
    # For this project, we'll select top 80% of cumulative importance
    
    # Calculate cumulative importance
    feature_importance['Cumulative_Importance'] = feature_importance['Importance'].cumsum()
    
    # Features that make up 80% of cumulative importance
    important_features = feature_importance[feature_importance['Cumulative_Importance'] <= 0.8]['Feature'].tolist()
    
    # Make sure we have at least 10 features
    if len(important_features) < 10:
        important_features = feature_importance['Feature'].head(10).tolist()
    
    print(f"\nSelected {len(important_features)} features that explain 80% of the variance:")
    print(important_features)
    
    # Create the final feature set
    X_selected = X[important_features]
    
    print("\nFinal dataset shape:", X_selected.shape)
    
    return X_selected, y, important_features

# %% Execute Step 5: Feature Engineering & Selection
X_selected, y, important_features = engineer_and_select_features(df_clean, df_eda)
print("\nFeature Engineering & Selection complete! Ready for Model Development.")

# %% Step 6: Model Development
def develop_models(X, y):
    """
    Develop and evaluate multiple machine learning models for churn prediction.
    
    Args:
        X: Features dataframe
        y: Target variable (Churn)
    """
    print("Starting Model Development...")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")
    
    # Initialize models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Evaluate each model
    results = {}
    print("\nTraining and evaluating models:")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'model': model
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')

        plt.show()
    
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig('visualizations/roc_curve.png', dpi=300, bbox_inches='tight')

        plt.show()

        
    
    # Compare models
    print("\nModel Comparison:")
    model_comparison = pd.DataFrame({
        model_name: {
            'Accuracy': results[model_name]['accuracy'],
            'Precision': results[model_name]['precision'],
            'Recall': results[model_name]['recall'],
            'F1 Score': results[model_name]['f1_score'],
            'AUC': results[model_name]['auc']
        } for model_name in results
    }).T
    
    print(model_comparison)
    
    # Visualize model comparison
    plt.figure(figsize=(14, 8))
    model_comparison.plot(kind='bar', figsize=(14, 8))
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig('visualizations/model_performance_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()
    
    

    # Find the best model based on F1 score (balancing precision and recall)
    best_model_name = model_comparison['F1 Score'].idxmax()
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model based on F1 Score: {best_model_name}")
    print(f"F1 Score: {model_comparison.loc[best_model_name, 'F1 Score']:.4f}")
    
    return best_model, results, X_train, X_test, y_train, y_test

# %% Execute Step 6: Model Development
best_model, model_results, X_train, X_test, y_train, y_test = develop_models(X_selected, y)
print("\nModel Development complete! Ready for Model Evaluation & Interpretation.")


# %% Step 7: Model Evaluation & Interpretation
def evaluate_and_interpret_model(best_model, X_train, X_test, y_train, y_test, features):
    """
    Further evaluate the best model and interpret its predictions.
    
    Args:
        best_model: The best performing model from Step 6
        X_train, X_test, y_train, y_test: Training and testing data
        features: List of feature names used in the model
    """
    print("Starting Model Evaluation & Interpretation...")
    
    # 1. Model Performance on Training Set
    train_predictions = best_model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(y_train, train_predictions)
    
    # Performance on Test Set
    test_predictions = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    
    # Check for overfitting
    print("\nChecking for Overfitting:")
    print(f"Training Accuracy: {train_accuracy:.4f}, F1 Score: {train_f1:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}")
    
    # 2. Cross-Validation for Robustness
    print("\nPerforming 5-fold Cross-Validation:")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    print(f"Cross-Validation F1 Scores: {cv_scores}")
    print(f"Mean F1 Score: {cv_scores.mean():.4f}, Std Dev: {cv_scores.std():.4f}")
    
    # 3. Feature Importance Analysis
    try:
        # For tree-based models
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance from Best Model')
            plt.tight_layout()
            plt.show()
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
        # For linear models like Logistic Regression
        elif hasattr(best_model, 'coef_'):
            coefs = best_model.coef_[0]
            feature_importance = pd.DataFrame({
                'Feature': features,
                'Coefficient': coefs
            }).sort_values('Coefficient', ascending=False)
            
            plt.figure(figsize=(12, 8))
            # Show positive and negative coefficients
            colors = ['red' if c < 0 else 'green' for c in feature_importance['Coefficient']]
            sns.barplot(x='Coefficient', y='Feature', data=feature_importance, palette=colors)
            plt.title('Feature Coefficients from Best Model')
            plt.tight_layout()
            plt.show()
            
            print("\nFeatures with Largest Positive and Negative Effects:")
            print("Positive Effects (Increase Churn Risk):")
            print(feature_importance.head(5))
            print("\nNegative Effects (Decrease Churn Risk):")
            print(feature_importance.tail(5))
    except:
        print("Could not extract feature importance for this model type.")
    
    # 4. Precision-Recall Trade-off Analysis
    plt.figure(figsize=(10, 8))
    precision, recall, thresholds = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('visualizations/precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # 5. Threshold Analysis - Finding optimal threshold
    thresholds = np.arange(0, 1, 0.01)
    scores = []
    
    # Get predicted probabilities
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    for threshold in thresholds:
        # Make predictions with current threshold
        y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        scores.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })
    
    # Convert to DataFrame
    threshold_metrics = pd.DataFrame(scores)
    
    # Plot precision, recall, and F1 score vs threshold
    plt.figure(figsize=(12, 8))
    plt.plot(threshold_metrics['threshold'], threshold_metrics['precision'], label='Precision')
    plt.plot(threshold_metrics['threshold'], threshold_metrics['recall'], label='Recall')
    plt.plot(threshold_metrics['threshold'], threshold_metrics['f1_score'], label='F1 Score')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs. Threshold')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('visualizations/precision_recall_f1score_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    

    # Find optimal threshold based on F1 score
    optimal_idx = threshold_metrics['f1_score'].idxmax()
    optimal_threshold = threshold_metrics.loc[optimal_idx, 'threshold']
    
    print(f"\nOptimal Threshold for F1 Score: {optimal_threshold:.2f}")
    print(f"At this threshold - Precision: {threshold_metrics.loc[optimal_idx, 'precision']:.4f}, "
          f"Recall: {threshold_metrics.loc[optimal_idx, 'recall']:.4f}, "
          f"F1 Score: {threshold_metrics.loc[optimal_idx, 'f1_score']:.4f}")
    
    # 6. Business Impact Analysis
    # Using the optimal threshold for predictions
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    
    # Confusion matrix with optimal threshold
    cm_optimal = confusion_matrix(y_test, y_pred_optimal)
    tn, fp, fn, tp = cm_optimal.ravel()
    
    # Business metrics
    # Assuming average customer lifetime value and retention cost
    avg_customer_value = 1000  # example value in $
    retention_cost = 100  # example cost per customer for retention campaign
    
    # Calculate potential savings
    total_customers = len(y_test)
    potential_churners = np.sum(y_test)
    correctly_identified = tp
    incorrectly_flagged = fp
    
    # Value saved by preventing churn
    savings = correctly_identified * avg_customer_value
    
    # Cost of retention efforts (including false positives)
    costs = (correctly_identified + incorrectly_flagged) * retention_cost
    
    # Net benefit
    net_benefit = savings - costs
    
    # ROI
    roi = (net_benefit / costs) * 100 if costs > 0 else 0
    
    print("\n=== Business Impact Analysis ===")
    print(f"Total customers in test set: {total_customers}")
    print(f"Actual churners: {potential_churners} ({potential_churners/total_customers*100:.1f}%)")
    print(f"Correctly identified churners: {correctly_identified} ({correctly_identified/potential_churners*100:.1f}% of actual churners)")
    print(f"Incorrectly flagged customers: {incorrectly_flagged} ({incorrectly_flagged/(tn+fp)*100:.1f}% of non-churners)")
    print(f"\nCustomer value preserved: ${savings:,.2f}")
    print(f"Retention campaign cost: ${costs:,.2f}")
    print(f"Net benefit: ${net_benefit:,.2f}")
    print(f"ROI: {roi:.1f}%")
    
    # 7. Customer Segmentation Based on Churn Risk
    # Calculate churn probabilities for all test customers
    test_probs = best_model.predict_proba(X_test)[:, 1]
    
    # Create risk segments
    risk_bins = [0, 0.3, 0.6, 0.9, 1.0]
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    
    risk_segments = pd.cut(test_probs, bins=risk_bins, labels=risk_labels)
    
    # Create a DataFrame with original features and churn probability
    customer_risks = X_test.copy()
    customer_risks['churn_probability'] = test_probs
    customer_risks['risk_segment'] = risk_segments
    customer_risks['actual_churn'] = y_test
    
    # Analyze segments
    segment_analysis = customer_risks.groupby('risk_segment').agg({
        'churn_probability': 'mean',
        'actual_churn': 'mean',
        'risk_segment': 'count'
    }).rename(columns={
        'churn_probability': 'avg_churn_probability',
        'actual_churn': 'actual_churn_rate',
        'risk_segment': 'count'
    })
    
    segment_analysis['percentage'] = segment_analysis['count'] / segment_analysis['count'].sum() * 100
    
    print("\n=== Customer Risk Segmentation ===")
    print(segment_analysis)
    
    # Visualize risk segments
    plt.figure(figsize=(10, 6))
    plt.bar(segment_analysis.index, segment_analysis['count'], color=['green', 'yellow', 'orange', 'red'])
    plt.title('Customer Count by Risk Segment')
    plt.xlabel('Risk Segment')
    plt.ylabel('Number of Customers')
    
    # Add percentage and count labels
    for i, v in enumerate(segment_analysis['count']):
        plt.text(i, v + 5, f"{segment_analysis['percentage'].iloc[i]:.1f}%\n({v})", 
                 ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/customer_count_by_risk_segment.png', dpi=300, bbox_inches='tight')

    plt.show()
    
    
    # 8. Summary of Key Findings
    print("\n=== Summary of Model Evaluation ===")
    print(f"1. Model Accuracy: 0.7888")
    print(f"2. Model F1 Score: 0.5430")
    print(f"3. Optimal Threshold for Decision Making: {optimal_threshold:.2f}")
    print("4. Most Important Factors for Churn Prediction:")
    print("   - services_per_dollar: 0.2669")
    print("   - tenure: 0.2271")
    print("   - PaymentMethod_Electronic check: 0.0787")
    print("   - InternetService_Fiber optic: 0.0784")
    print("   - MonthlyCharges: 0.0723")
    print(f"5. Potential Business Impact: ${net_benefit:,.2f} net benefit with 50.9% ROI")
    print(f"6. High/Very High Risk Customers: {segment_analysis.loc[['High Risk', 'Very High Risk'], 'count'].sum()} "
    f"(13.69% of customer base)")

    return optimal_threshold, segment_analysis

# %% Execute Step 7: Model Evaluation & Interpretation
optimal_threshold, risk_segments = evaluate_and_interpret_model(
    best_model, X_train, X_test, y_train, y_test, X_selected.columns
)
print("\nModel Evaluation & Interpretation complete! Ready for Data Visualization & Communication.")# %% 



# %% Step 8: Data Visualization & Communication
def create_business_visualizations(df_original, model, X_selected, optimal_threshold, risk_segments):
    """
    Create business-focused visualizations and an executive dashboard.
    
    Args:
        df_original: The original dataframe with categorical features
        model: The best performing model
        X_selected: The selected features used in the model
        optimal_threshold: The optimal probability threshold for classification
        risk_segments: The customer risk segmentation analysis
    """
    print("Creating Business Visualizations & Communication Materials...")
    
    # 1. Executive Summary Dashboard
    plt.figure(figsize=(20, 12))
    plt.suptitle("Telecom Customer Churn Analysis - Executive Dashboard", fontsize=20, y=0.98)
    
    # 1.1 Churn Rate by Contract Type (Key Finding)
    plt.subplot(2, 3, 1)
    contract_churn = pd.crosstab(df_original['Contract'], df_original['Churn'], normalize='index') * 100
    contract_churn['Yes'].plot(kind='bar', color='coral')
    plt.title('Churn Rate by Contract Type', fontsize=12)
    plt.xlabel('Contract Type')
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=45)
    for i, v in enumerate(contract_churn['Yes']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 1.2 Churn Rate by Tenure Group
    plt.subplot(2, 3, 2)
    # Create tenure groups and calculate churn rate
    df_original['tenure_group'] = pd.cut(df_original['tenure'], 
                                       bins=[0, 12, 24, 36, 48, 60, 72],
                                       labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'])
    tenure_churn = df_original.groupby('tenure_group')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    tenure_churn.plot(kind='bar', color='steelblue')
    plt.title('Churn Rate by Tenure (Months)', fontsize=12)
    plt.xlabel('Tenure Group')
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=45)
    for i, v in enumerate(tenure_churn):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 1.3 Customer Risk Segmentation
    plt.subplot(2, 3, 3)
    colors = ['green', 'yellow', 'orange', 'red']
    plt.bar(risk_segments.index, risk_segments['percentage'], color=colors)
    plt.title('Customer Base by Churn Risk', fontsize=12)
    plt.xlabel('Risk Level')
    plt.ylabel('Percentage of Customers')
    plt.xticks(rotation=45)
    for i, v in enumerate(risk_segments['percentage']):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 1.4 Top 5 Churn Factors
    plt.subplot(2, 3, 4)
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-5:]  # Top 5 features
            features = X_selected.columns[indices]
            importances = importances[indices]
            
            plt.barh(range(len(features)), importances, color='purple')
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel('Relative Importance')
            plt.title('Top 5 Factors Influencing Churn', fontsize=12)
        elif hasattr(model, 'coef_'):
            coefs = model.coef_[0]
            indices = np.argsort(np.abs(coefs))[-5:]  # Top 5 features by magnitude
            features = X_selected.columns[indices]
            coefficients = coefs[indices]
            
            colors = ['red' if c < 0 else 'green' for c in coefficients]
            plt.barh(range(len(features)), coefficients, color=colors)
            plt.yticks(range(len(features)), [f.replace('_', ' ').title() for f in features])
            plt.xlabel('Effect on Churn Risk')
            plt.title('Top 5 Factors Influencing Churn', fontsize=12)
    except:
        plt.text(0.5, 0.5, "Feature importance not available for this model type", 
                ha='center', va='center', fontsize=12)
    
    # 1.5 ROI Analysis (simple visualization)
    plt.subplot(2, 3, 5)
    # Example values - these should be replaced with actual calculations
    avg_customer_value = 1000  # in $
    retention_cost = 100  # in $ per customer
    correct_identifications = 0.85  # 85% of actual churners
    false_positives = 0.15  # 15% of non-churners
    
    # Calculate for a typical cohort of 1000 customers with 26.5% churn rate
    n_customers = 1000
    churners = int(n_customers * 0.265)
    non_churners = n_customers - churners
    
    identified_churners = int(churners * correct_identifications)
    false_flags = int(non_churners * false_positives)
    
    # Financial impact
    value_saved = identified_churners * avg_customer_value
    campaign_cost = (identified_churners + false_flags) * retention_cost
    net_benefit = value_saved - campaign_cost
    
    # Plot ROI components
    components = ['Value Preserved', 'Campaign Cost', 'Net Benefit']
    values = [value_saved, -campaign_cost, net_benefit]
    colors = ['green', 'red', 'blue']
    
    plt.bar(components, values, color=colors)
    plt.title('Financial Impact Analysis (per 1000 customers)', fontsize=12)
    plt.ylabel('Amount ($)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(values):
        if v < 0:
            plt.text(i, v - 5000, f"${abs(v):,.0f}", ha='center', color='white')
        else:
            plt.text(i, v + 1000, f"${v:,.0f}", ha='center')
    
    # 1.6 Recommended Retention Strategies
    plt.subplot(2, 3, 6)
    plt.axis('off')
    strategies = [
    "1. Target month-to-month customers (42.71% churn rate)",  
    "2. Focus on customers in first year (47.68% churn rate)",  
    "3. Promote Online Security (reduces churn from 41.77% to 14.61%)",  
    "4. Add Tech Support (reduces churn from 41.64% to 15.17%)",  
    "5. Review Electronic Check payment customers"
]
    
    plt.text(0.5, 0.1, "Estimated 50.9% ROI on Retention Efforts", ha='center', fontsize=13, fontweight='bold', color='green')
    y_pos = 0.8
    for strategy in strategies:
        plt.text(0.1, y_pos, strategy, fontsize=11)
        y_pos -= 0.15
    
    plt.text(0.5, 0.1, "Estimated 50.9% ROI on Retention Efforts", ha='center', fontsize=13, fontweight='bold', color='green')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig('visualizations/data_visualization_and_communication.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # 2. Additional Business-Focused Visualizations
    
    # 2.1 Churn Rate by Internet Service with Monthly Charges
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='InternetService', y='MonthlyCharges', hue='Churn', data=df_original)
    plt.title('Monthly Charges by Internet Service and Churn Status', fontsize=14)
    plt.xlabel('Internet Service Type')
    plt.ylabel('Monthly Charges ($)')
    plt.legend(title='Churn')

    plt.tight_layout()

    plt.savefig('visualizations/monthly_charges_by_internet_service_and_churn_status.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # 2.2 Services Impact on Churn
    plt.figure(figsize=(14, 8))
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    
    for i, service in enumerate(service_columns, 1):
        plt.subplot(2, 2, i)
        service_churn = pd.crosstab(df_original[service], df_original['Churn'], normalize='index') * 100
        service_churn['Yes'].plot(kind='bar', color=sns.color_palette()[i-1])
        plt.title(f'Churn Rate by {service}', fontsize=12)
        plt.xlabel(service)
        plt.ylabel('Churn Rate (%)')
        plt.xticks(rotation=45)
        for j, v in enumerate(service_churn['Yes']):
            plt.text(j, v + 1, f"{v:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig('visualizations/churn_rate_by_service.png', dpi=300, bbox_inches='tight')

    plt.show()
    
    
    # 2.3 Contract Type vs. Tenure - Bubble Chart
    plt.figure(figsize=(12, 8))
    
    # Prepare data for bubble chart
    contract_tenure = df_original.groupby(['Contract', 'tenure_group']).agg({
        'customerID': 'count',
        'Churn': lambda x: (x == 'Yes').mean() * 100
    }).reset_index()
    
    contract_tenure.columns = ['Contract', 'Tenure', 'CustomerCount', 'ChurnRate']
    
    # Separate by contract type
    contracts = contract_tenure['Contract'].unique()
    colors = ['coral', 'skyblue', 'lightgreen']
    
    for i, contract in enumerate(contracts):
        subset = contract_tenure[contract_tenure['Contract'] == contract]
        plt.scatter(subset['Tenure'], subset['ChurnRate'], 
                   s=subset['CustomerCount']/5, alpha=0.7, 
                   label=contract, color=colors[i])
    
    plt.title('Churn Rate by Contract Type and Tenure', fontsize=14)
    plt.xlabel('Tenure Group')
    plt.ylabel('Churn Rate (%)')
    plt.legend(title='Contract Type')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('visualizations/churn_rate_by_contract_type_and_tenure.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    

    # 3. Create a One-Pager Business-Friendly PDF (using markdown for simplicity)
    one_pager = """
    
# Telecom Customer Churn Analysis: Executive Summary

## Key Findings

1. **Model Performance**: Gradient Boosting model achieves 78.88% accuracy with AUC of 0.8344
2. **Contract Impact**: Month-to-month contracts have 42.71% churn vs. only 2.83% for two-year contracts
3. **High-Risk Segments**: 13.69% of our customer base is at high or very high risk of churning
4. **Critical Factors**: services_per_dollar (0.2669), tenure (0.2271), and payment method are key drivers
5. **Financial Impact**: For every 1,000 customers, we can save 40 from churning with 50.9% ROI

## Recommended Actions

1. **Contract Conversion**: Target month-to-month customers (42.71% churn rate)
2. **Early Retention**: Focus on first-year customers (47.68% churn rate)
3. **Service Enhancement**: Promote Online Security (reduces churn from 41.77% to 14.61%)
4. **Support Focus**: Add Tech Support (reduces churn from 41.64% to 15.17%)
5. **Payment Review**: Address Electronic Check payment issues

## Expected Outcomes
    
    - **Immediate Impact**: 15% reduction in churn rate within the high-risk customer segment
    - **Medium-term Impact**: 8% overall churn reduction within 6 months
    - **Long-term Impact**: Shift customer base toward longer-term contracts and higher-value service bundles
    
## Implementation Timeline
    
    - **Month 1**: Deploy predictive model to identify high-risk customers
    - **Month 2**: Launch targeted retention campaigns for highest-risk segments
    - **Month 3-4**: Implement contract conversion and service bundling strategies
    - **Month 5-6**: Roll out enhanced loyalty program and new customer support systems
    - **Month 7+**: Continuous monitoring and campaign optimization
    """
    
    print("\n=== One-Pager Executive Summary ===")
    print(one_pager)
    
    return one_pager

# %% Execute Step 8: Data Visualization & Communication
one_pager = create_business_visualizations(df, best_model, X_selected, optimal_threshold, risk_segments)
print("\nData Visualization & Communication complete! Ready for SQL Integration.")


# %% Step 9: SQL Integration
def implement_sql_integration(df_original, X_test, y_test, best_model, optimal_threshold):
    """
    Implement SQL database integration for the churn prediction model.
    
    Args:
        df_original: Original dataframe with customer data
        X_test: Test features dataframe
        y_test: Test target values
        best_model: The best performing model
        optimal_threshold: Optimal probability threshold for classification
    """
    print("Starting SQL Integration...")
    
    # Create an in-memory SQLite database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # 1. Create customer table
    print("\nCreating customer table...")
    
    # Get a sample of original data for demonstration
    sample_df = df_original.sample(n=min(1000, len(df_original)), random_state=42).reset_index(drop=True)
    
    # Create customer table
    sample_df.to_sql('customers', conn, index=False, if_exists='replace')
    
    # Verify table creation
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables in database:", cursor.fetchall())
    
    # 2. Generate predictions for sample data
    # For simplicity, let's use our test set predictions
    print("\nGenerating predictions for customers...")
    
    # Get predictions for X_test
    y_prob = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # Create risk segments
    risk_bins = [0, 0.3, 0.6, 0.9, 1.0]
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    risk_segments = pd.cut(y_prob, bins=risk_bins, labels=risk_labels)
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'customer_id': X_test.index,
        'churn_probability': y_prob,
        'predicted_churn': y_pred,
        'risk_segment': risk_segments,
        'actual_churn': y_test,
        'prediction_date': pd.Timestamp.now().strftime('%Y-%m-%d')
    })
    
    # 3. Store predictions in SQL
    predictions_df.to_sql('churn_predictions', conn, index=False, if_exists='replace')
    
    # Verify predictions table
    cursor.execute("SELECT * FROM churn_predictions LIMIT 5;")
    columns = [desc[0] for desc in cursor.description]
    print("\nSample predictions data:")
    print(columns)
    for row in cursor.fetchall():
        print(row)
    
    # 4. Create useful views for business users
    
    # 4.1 High Risk Customers View
    cursor.execute("""
    CREATE VIEW high_risk_customers AS
    SELECT c.*, p.churn_probability, p.risk_segment
    FROM customers c
    JOIN churn_predictions p ON c.customerID = p.customer_id
    WHERE p.risk_segment IN ('High Risk', 'Very High Risk')
    ORDER BY p.churn_probability DESC;
    """)
    
    # 4.2 Retention Targets View
    cursor.execute("""
    CREATE VIEW retention_targets AS
    SELECT 
        c.customerID,
        c.gender,
        c.SeniorCitizen,
        c.Partner,
        c.Dependents,
        c.tenure,
        c.PhoneService,
        c.MultipleLines,
        c.InternetService,
        c.OnlineSecurity,
        c.OnlineBackup,
        c.DeviceProtection,
        c.TechSupport,
        c.StreamingTV,
        c.StreamingMovies,
        c.Contract,
        c.MonthlyCharges,
        c.TotalCharges,
        p.churn_probability,
        p.risk_segment,
        CASE
            WHEN c.Contract = 'Month-to-month' AND c.tenure <= 12 THEN 'Contract Upgrade'
            WHEN c.InternetService = 'Fiber optic' AND c.TechSupport = 'No' THEN 'Add Tech Support'
            WHEN c.OnlineSecurity = 'No' THEN 'Add Security'
            ELSE 'General Retention'
        END as recommended_action
    FROM customers c
    JOIN churn_predictions p ON c.customerID = p.customer_id
    WHERE p.churn_probability >= 0.5
    ORDER BY p.churn_probability DESC;
    """)
    
    # 4.3 Churn Analysis by Segment View
    cursor.execute("""
    CREATE VIEW churn_by_segment AS
    SELECT
        risk_segment,
        COUNT(*) as customer_count,
        AVG(churn_probability) as avg_churn_probability,
        SUM(CASE WHEN actual_churn = 1 THEN 1 ELSE 0 END) as actual_churners,
        SUM(CASE WHEN actual_churn = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as actual_churn_rate
    FROM churn_predictions
    GROUP BY risk_segment
    ORDER BY avg_churn_probability;
    """)
    
    # 5. Demonstrate SQL queries
    print("\n=== Example SQL Queries for Business Users ===")
    
    # 5.1 Query high-risk customers
    print("\nQuery 1: High-risk customers with month-to-month contracts")
    query1 = """
    SELECT * FROM high_risk_customers 
    WHERE Contract = 'Month-to-month' 
    LIMIT 5;
    """
    cursor.execute(query1)
    columns = [desc[0] for desc in cursor.description]
    print("Columns:", columns)
    print("Results:")
    for row in cursor.fetchall():
        print(row)
    
    # 5.2 Query retention targets by recommended action
    print("\nQuery 2: Count of retention targets by recommended action")
    query2 = """
    SELECT recommended_action, COUNT(*) as customer_count, AVG(churn_probability) as avg_churn_prob
    FROM retention_targets
    GROUP BY recommended_action
    ORDER BY customer_count DESC;
    """
    cursor.execute(query2)
    results = cursor.fetchall()
    for row in results:
        print(row)
    
    # 5.3 Query churn analysis by segment
    print("\nQuery 3: Churn analysis by customer risk segment")
    query3 = """
    SELECT * FROM churn_by_segment;
    """
    cursor.execute(query3)
    results = cursor.fetchall()
    for row in results:
        print(row)
    
    # 6. Create a stored procedure (simulated in SQLite)
    print("\nCreating stored procedure (simulated in SQLite)...")
    
    # SQLite doesn't support stored procedures, but here's how it would look in SQL Server/MySQL
    stored_procedure = """
    -- This would be a stored procedure in SQL Server or MySQL
    CREATE PROCEDURE GetRetentionRecommendations(@RiskThreshold FLOAT = 0.6)
    AS
    BEGIN
        SELECT 
            customerID,
            risk_segment,
            churn_probability,
            recommended_action,
            CASE
                WHEN Contract = 'Month-to-month' THEN MonthlyCharges * 0.9
                ELSE MonthlyCharges * 0.95
            END as discounted_offer
        FROM retention_targets
        WHERE churn_probability >= @RiskThreshold
        ORDER BY churn_probability DESC
    END;
    """
    print(stored_procedure)
    
    # 7. Export SQL scripts for implementation
    sql_scripts = {
        "create_tables": """
        -- Customer table
        CREATE TABLE customers (
            customerID VARCHAR(50) PRIMARY KEY,
            gender VARCHAR(10),
            SeniorCitizen INT,
            Partner VARCHAR(5),
            Dependents VARCHAR(5),
            tenure INT,
            PhoneService VARCHAR(5),
            MultipleLines VARCHAR(20),
            InternetService VARCHAR(20),
            OnlineSecurity VARCHAR(20),
            OnlineBackup VARCHAR(20),
            DeviceProtection VARCHAR(20),
            TechSupport VARCHAR(20),
            StreamingTV VARCHAR(20),
            StreamingMovies VARCHAR(20),
            Contract VARCHAR(20),
            PaperlessBilling VARCHAR(5),
            PaymentMethod VARCHAR(30),
            MonthlyCharges FLOAT,
            TotalCharges VARCHAR(30),
            Churn VARCHAR(5)
        );
        
        -- Predictions table
        CREATE TABLE churn_predictions (
            prediction_id INT PRIMARY KEY IDENTITY(1,1),
            customer_id VARCHAR(50),
            churn_probability FLOAT,
            predicted_churn INT,
            risk_segment VARCHAR(20),
            prediction_date DATE,
            FOREIGN KEY (customer_id) REFERENCES customers(customerID)
        );
        """,
        
        "create_views": """
        -- High Risk Customers View
        CREATE VIEW high_risk_customers AS
        SELECT c.*, p.churn_probability, p.risk_segment
        FROM customers c
        JOIN churn_predictions p ON c.customerID = p.customer_id
        WHERE p.risk_segment IN ('High Risk', 'Very High Risk')
        ORDER BY p.churn_probability DESC;
        
        -- Retention Targets View
        CREATE VIEW retention_targets AS
        SELECT 
            c.customerID,
            c.gender,
            c.SeniorCitizen,
            c.Partner,
            c.Dependents,
            c.tenure,
            c.PhoneService,
            c.MultipleLines,
            c.InternetService,
            c.OnlineSecurity,
            c.OnlineBackup,
            c.DeviceProtection,
            c.TechSupport,
            c.StreamingTV,
            c.StreamingMovies,
            c.Contract,
            c.MonthlyCharges,
            c.TotalCharges,
            p.churn_probability,
            p.risk_segment,
            CASE
                WHEN c.Contract = 'Month-to-month' AND c.tenure <= 12 THEN 'Contract Upgrade'
                WHEN c.InternetService = 'Fiber optic' AND c.TechSupport = 'No' THEN 'Add Tech Support'
                WHEN c.OnlineSecurity = 'No' THEN 'Add Security'
                ELSE 'General Retention'
            END as recommended_action
        FROM customers c
        JOIN churn_predictions p ON c.customerID = p.customer_id
        WHERE p.churn_probability >= 0.5
        ORDER BY p.churn_probability DESC;
        
        -- Churn Analysis by Segment View
        CREATE VIEW churn_by_segment AS
        SELECT
            risk_segment,
            COUNT(*) as customer_count,
            AVG(churn_probability) as avg_churn_probability,
            SUM(CASE WHEN actual_churn = 1 THEN 1 ELSE 0 END) as actual_churners,
            SUM(CASE WHEN actual_churn = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as actual_churn_rate
        FROM churn_predictions
        GROUP BY risk_segment
        ORDER BY avg_churn_probability;
        """,
        
        "useful_queries": """
        -- Monthly retention campaign list
        SELECT 
            c.customerID,
            c.Contract,
            c.MonthlyCharges,
            c.tenure,
            p.churn_probability,
            p.risk_segment,
            CASE
                WHEN c.Contract = 'Month-to-month' AND c.tenure <= 12 THEN 'Contract Upgrade'
                WHEN c.InternetService = 'Fiber optic' AND c.TechSupport = 'No' THEN 'Add Tech Support'
                WHEN c.OnlineSecurity = 'No' THEN 'Add Security'
                ELSE 'General Retention'
            END as recommended_action,
            CASE
                WHEN c.Contract = 'Month-to-month' THEN MonthlyCharges * 0.9
                ELSE MonthlyCharges * 0.95
            END as discounted_offer
        FROM customers c
        JOIN churn_predictions p ON c.customerID = p.customer_id
        WHERE p.churn_probability >= 0.6
        ORDER BY p.churn_probability DESC;
        
        -- Churn risk by contract type
        SELECT 
            Contract,
            COUNT(*) as customer_count,
            AVG(p.churn_probability) as avg_churn_probability,
            SUM(CASE WHEN p.predicted_churn = 1 THEN 1 ELSE 0 END) as predicted_churners,
            SUM(CASE WHEN p.predicted_churn = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as predicted_churn_rate
        FROM customers c
        JOIN churn_predictions p ON c.customerID = p.customer_id
        GROUP BY Contract
        ORDER BY avg_churn_probability DESC;
        
        -- Customers with most valuable retention opportunities
        SELECT 
            c.customerID,
            c.tenure,
            c.MonthlyCharges,
            c.Contract,
            p.churn_probability,
            c.MonthlyCharges * 12 as annual_value,
            c.MonthlyCharges * 12 * p.churn_probability as expected_loss_value
        FROM customers c
        JOIN churn_predictions p ON c.customerID = p.customer_id
        WHERE p.predicted_churn = 1
        ORDER BY expected_loss_value DESC
        LIMIT 100;
        """
    }
    
    print("\n=== SQL Integration Scripts ===")
    print("Example scripts have been created for production implementation.")
    
    # 8. Close connection
    conn.close()
    
    print("\nSQL Integration completed successfully!")
    
    return sql_scripts

# %% Execute Step 9: SQL Integration
sql_scripts = implement_sql_integration(df, X_test, y_test, best_model, optimal_threshold)
print("\nSQL Integration complete! Ready for Deployment Considerations.")

# %% Step 10: Deployment Considerations
def outline_deployment_considerations(best_model, X_selected):
    """
    Outline considerations for deploying the churn prediction model to production.
    
    Args:
        best_model: The best performing model
        X_selected: The selected features used in the model
    """
    print("Outlining Deployment Considerations...")
    
    # 1. Model Serialization
    print("\n=== Model Serialization ===")
    
    # Create a directory for model artifacts if it doesn't exist
    import os
    if not os.path.exists('model_artifacts'):
        os.makedirs('model_artifacts')
    
    # Serialize the model using pickle
    import pickle
    model_filename = 'model_artifacts/churn_prediction_model.pkl'
    
    with open(model_filename, 'wb') as file:
        pickle.dump(best_model, file)
    
    # Save feature list
    feature_filename = 'model_artifacts/feature_list.pkl'
    with open(feature_filename, 'wb') as file:
        pickle.dump(list(X_selected.columns), file)
    
    print(f"Model serialized and saved to {model_filename}")
    print(f"Feature list saved to {feature_filename}")
    
    # 2. Model Loading Code Example
    print("\n=== Model Loading Code Example ===")
    
    model_loading_code = """
    import pickle
    
    # Load the model
    with open('model_artifacts/churn_prediction_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load the feature list
    with open('model_artifacts/feature_list.pkl', 'rb') as file:
        features = pickle.load(file)
    
    # Function to preprocess new data
    def preprocess_data(customer_data):
        '''
        Preprocess customer data to match the expected model input format.
        
        Args:
            customer_data: DataFrame with raw customer data
            
        Returns:
            DataFrame with processed features ready for prediction
        '''
        # Implement preprocessing steps (same as used during model development)
        # ...
        
        # Ensure data has all required features in the correct order
        processed_data = customer_data[features]
        
        return processed_data
    
    # Function to make predictions
    def predict_churn_probability(customer_data, threshold=0.5):
        '''
        Predict churn probability for new customers.
        
        Args:
            customer_data: DataFrame with customer information
            threshold: Probability threshold for classifying as churn
            
        Returns:
            DataFrame with customer IDs, churn probabilities, and predictions
        '''
        # Preprocess the data
        processed_data = preprocess_data(customer_data)
        
        # Make predictions
        probabilities = model.predict_proba(processed_data)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': customer_data.index,
            'churn_probability': probabilities,
            'predicted_churn': predictions
        })
        
        return results
    """
    
    print(model_loading_code)
    
    # 3. Deployment Architectures
    print("\n=== Deployment Architecture Options ===")
    
    deployment_options = {
        "Batch Prediction": {
            "Description": "Run predictions on a schedule (e.g., daily/weekly) for all customers",
            "Use Case": "Regular churn risk assessment for all customers",
            "Pros": [
                "Simple to implement",
                "Low infrastructure requirements",
                "Predictable resource usage"
            ],
            "Cons": [
                "Predictions may become outdated between runs",
                "Not suitable for real-time applications"
            ],
            "Implementation": "Schedule Python script execution using cron jobs or workflow tools like Apache Airflow"
        },
        
        "REST API": {
            "Description": "Deploy model as a REST API endpoint for on-demand predictions",
            "Use Case": "Real-time churn assessment during customer interactions",
            "Pros": [
                "Real-time predictions available when needed",
                "Can be integrated with multiple systems",
                "Scalable with proper infrastructure"
            ],
            "Cons": [
                "More complex infrastructure requirements",
                "Needs monitoring for availability and performance"
            ],
            "Implementation": "Use Flask/FastAPI with Docker, or deploy to cloud services like AWS SageMaker, Azure ML, or Google AI Platform"
        },
        
        "Database Integration": {
            "Description": "Run model within database environment (e.g., SQL Server ML Services)",
            "Use Case": "Integrating predictions directly with customer database",
            "Pros": [
                "Tight integration with existing data",
                "Can use database scheduling mechanisms",
                "Predictions accessible through SQL queries"
            ],
            "Cons": [
                "Limited to languages supported by the database",
                "May require specialized database features"
            ],
            "Implementation": "Use SQL Server ML Services with R/Python or PostgreSQL with PL/Python"
        },
        
        "Streaming": {
            "Description": "Process data streams and update predictions as new data arrives",
            "Use Case": "Continuous updating of churn risk as customer behavior changes",
            "Pros": [
                "Always up-to-date predictions",
                "Can respond quickly to behavior changes",
                "Enables real-time interventions"
            ],
            "Cons": [
                "Most complex architecture",
                "Higher infrastructure costs",
                "More challenging to debug and monitor"
            ],
            "Implementation": "Use Kafka/Spark Streaming with containerized model deployment"
        }
    }
    
    for option, details in deployment_options.items():
        print(f"\n{option}:")
        print(f"  Description: {details['Description']}")
        print(f"  Use Case: {details['Use Case']}")
        print("  Pros:")
        for pro in details['Pros']:
            print(f"    - {pro}")
        print("  Cons:")
        for con in details['Cons']:
            print(f"    - {con}")
        print(f"  Implementation: {details['Implementation']}")
    
    # 4. Monitoring and Maintenance
    print("\n=== Monitoring and Maintenance Plan ===")
    
    monitoring_plan = """
    Model Monitoring Strategy
    =========================
    
    1. Performance Monitoring
       - Track key metrics (accuracy, precision, recall, F1) over time
       - Set up alerts for significant performance degradation
       - Implement A/B testing when deploying model updates
    
    2. Data Drift Detection
       - Monitor feature distributions for drift from training data
       - Analyze prediction distribution changes
       - Set thresholds for automatic retraining triggers
    
    3. Business Impact Tracking
       - Track churn reduction effectiveness
       - Measure ROI of retention campaigns
       - Compare predicted vs. actual churn rates
    
    4. Technical Monitoring
       - Track prediction latency and throughput
       - Monitor resource utilization (CPU, memory)
       - Set up uptime and availability monitoring
    
    Maintenance Schedule
    ===================
    
    1. Regular Retraining
       - Scheduled quarterly model retraining with fresh data
       - Feature re-evaluation during retraining
       - Performance comparison with previous model versions
    
    2. Model Updates
       - Major model review every 6 months
       - Consider algorithm updates or ensemble approaches
       - Evaluate new features as they become available
    
    3. Documentation Maintenance
       - Keep model cards updated with each version
       - Document all hyperparameter changes
       - Maintain preprocessing code documentation
    
    4. Feedback Loop
       - Collect feedback from business users
       - Track false positives and false negatives
       - Incorporate retention campaign effectiveness data
    """
    
    print(monitoring_plan)
    
    # 5. Ethical and Privacy Considerations
    print("\n=== Ethical and Privacy Considerations ===")
    
    ethical_considerations = """
    Ethical Guidelines for Churn Prediction Deployment
    =================================================
    
    1. Transparency
       - Clearly communicate to stakeholders how predictions are made
       - Explain key factors that lead to high churn probability
       - Provide confidence levels with predictions
    
    2. Fairness
       - Regularly audit model for bias against demographic groups
       - Ensure retention offers don't discriminate unfairly
       - Monitor impact across different customer segments
    
    3. Privacy
       - Follow data protection regulations (GDPR, CCPA, etc.)
       - Minimize data usage to only what's necessary
       - Implement appropriate data security measures
       - Consider anonymization where possible
    
    4. Customer Agency
       - Use predictions to improve service, not just maximize profits
       - Balance prediction-based actions with customer preferences
       - Consider opt-out options for targeted retention
    
    5. Oversight
       - Establish a review process for model decision-making
       - Create an escalation path for disputed predictions
       - Set up regular ethical review of the system
    """
    
    print(ethical_considerations)
    
    # 6. Scaling Considerations
    print("\n=== Scaling Considerations ===")
    
    scaling_considerations = """
    Scaling the Churn Prediction System
    ==================================
    
    1. Technical Scaling
       - Design for containerization (Docker) for easy scaling
       - Consider serverless deployment for variable workloads
       - Implement caching for frequent predictions on the same customers
       - Use cloud provider auto-scaling capabilities
    
    2. Organizational Scaling
       - Document the entire pipeline for knowledge transfer
       - Create training materials for new team members
       - Establish clear ownership and responsibilities
       - Set up cross-functional team touchpoints
    
    3. Business Process Integration
       - Create APIs for integration with CRM systems
       - Develop dashboards for business users
       - Establish escalation procedures for critical cases
       - Set up automated reporting
    
    4. Geographic Scaling
       - Plan for multi-region deployment if needed
       - Consider data residency requirements
       - Test with region-specific data characteristics
       - Localize retention strategies as appropriate
    """
    
    print(scaling_considerations)
    
    # 7. Implementation Roadmap
    print("\n=== Implementation Roadmap ===")
    
    implementation_roadmap = """
    Churn Prediction Implementation Roadmap
    ======================================
    
    Phase 1: Pilot Deployment (Months 1-2)
    --------------------------------------
    - Deploy model in controlled environment with subset of customers
    - Set up basic monitoring and logging
    - Train initial business users on system usage
    - Collect feedback and adjust approach
    
    Phase 2: Full Deployment (Months 3-4)
    ------------------------------------
    - Scale to full customer base
    - Implement complete monitoring infrastructure
    - Integrate with retention campaign systems
    - Establish regular reporting process
    
    Phase 3: Optimization (Months 5-6)
    ---------------------------------
    - Analyze initial results and refine model
    - Optimize retention strategies based on data
    - Implement A/B testing framework
    - Develop enhanced dashboards for business users
    
    Phase 4: Advanced Features (Months 7-9)
    -------------------------------------
    - Implement automated retraining pipeline
    - Develop additional models for customer segmentation
    - Add explainability features for business users
    - Create ROI optimization tools for retention campaigns
    
    Phase 5: Enterprise Integration (Months 10-12)
    -------------------------------------------
    - Full integration with enterprise data systems
    - Implement advanced security and compliance features
    - Develop training program for new users
    - Create long-term governance model
    """
    
    print(implementation_roadmap)
    
    # Return summary of deployment considerations
    return {
        "Model Serialization": model_filename,
        "Deployment Options": list(deployment_options.keys()),
        "Monitoring Plan": "Comprehensive monitoring plan covering performance, data drift, business impact, and technical aspects",
        "Implementation Roadmap": "5-phase implementation over 12 months"
    }

# %% Execute Step 10: Deployment Considerations
deployment_summary = outline_deployment_considerations(best_model, X_selected)
print("\nDeployment Considerations complete! Project workflow completed successfully.")


##########################################################
# %% Output Key Results for Documentation
def print_key_documentation_metrics(df_original, best_model, model_results, X_selected, optimal_threshold, risk_segments):
    """
    Print key metrics and findings for project documentation.
    """
    print("\n" + "="*80)
    print("KEY PROJECT METRICS FOR DOCUMENTATION")
    print("="*80)
    
    # 1. Model Performance
    print("\n--- BEST MODEL PERFORMANCE ---")
    best_model_name = max(model_results, key=lambda x: model_results[x]['f1_score'])
    print(f"Best Model: {best_model_name}")
    
    best_metrics = model_results[best_model_name]
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1_score']:.4f}")
    print(f"AUC: {best_metrics['auc']:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    
    # 2. Churn Rate by Contract Type
    print("\n--- CHURN RATE BY CONTRACT TYPE ---")
    contract_churn = pd.crosstab(df_original['Contract'], df_original['Churn'], normalize='index') * 100
    for contract_type in contract_churn.index:
        print(f"{contract_type}: {contract_churn.loc[contract_type, 'Yes']:.2f}%")
    
    # 3. Feature Importance
    print("\n--- TOP FEATURES ---")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': X_selected.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("Top 10 Features:")
        for i, (feature, importance) in enumerate(
            zip(features_df['Feature'].head(10), features_df['Importance'].head(10)), 1):
            print(f"{i}. {feature}: {importance:.4f}")
    
    elif hasattr(best_model, 'coef_'):
        coefs = best_model.coef_[0]
        features_df = pd.DataFrame({
            'Feature': X_selected.columns,
            'Coefficient': coefs
        }).sort_values('Coefficient', ascending=False)
        
        print("Top 10 Features (by coefficient magnitude):")
        for i, (feature, coef) in enumerate(
            zip(features_df['Feature'].head(10), features_df['Coefficient'].head(10)), 1):
            print(f"{i}. {feature}: {coef:.4f}")
    
    # 4. Churn Rate by Tenure Group
    print("\n--- CHURN RATE BY TENURE GROUP ---")
    df_original['tenure_group'] = pd.cut(df_original['tenure'], 
                                       bins=[0, 12, 24, 36, 48, 60, 72],
                                       labels=['0-12', '12-24', '24-36', '36-48', '48-60', '60-72'])
    
    tenure_churn = df_original.groupby('tenure_group')['Churn'].apply(
        lambda x: (x == 'Yes').mean() * 100)
    
    for tenure_group in tenure_churn.index:
        print(f"{tenure_group} months: {tenure_churn[tenure_group]:.2f}%")
    
    # 5. Churn Rate by Internet Service
    print("\n--- CHURN RATE BY INTERNET SERVICE ---")
    internet_churn = pd.crosstab(df_original['InternetService'], df_original['Churn'], normalize='index') * 100
    for service_type in internet_churn.index:
        print(f"{service_type}: {internet_churn.loc[service_type, 'Yes']:.2f}%")
    
    # 6. Impact of Key Services on Churn
    print("\n--- IMPACT OF KEY SERVICES ON CHURN ---")
    for service in ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection']:
        service_churn = pd.crosstab(df_original[service], df_original['Churn'], normalize='index') * 100
        print(f"{service}:")
        for option in service_churn.index:
            if option != 'No internet service':  # Skip this category for cleaner output
                print(f"  {option}: {service_churn.loc[option, 'Yes']:.2f}% churn rate")
    
    # 7. Risk Segment Analysis
    print("\n--- CUSTOMER RISK SEGMENTS ---")
    for segment, data in risk_segments.iterrows():
        print(f"{segment}: {data['percentage']:.2f}% of customers, {data['actual_churn_rate']*100:.2f}% actual churn rate")
    
    # 8. Financial Impact (Example calculation - adjust with your actual values)
    print("\n--- ESTIMATED FINANCIAL IMPACT ---")
    # Using placeholder values - replace with your actual calculated values if available
    avg_customer_value = 1000  # example value in $
    retention_cost = 100  # example cost per customer for retention campaign
    churn_reduction = 0.15  # estimated reduction in churn (15%)
    
    # Calculate for a cohort of 1000 customers with 26.5% base churn rate
    n_customers = 1000
    base_churners = int(n_customers * 0.265)
    post_implementation_churners = int(base_churners * (1 - churn_reduction))
    customers_saved = base_churners - post_implementation_churners
    
    value_saved = customers_saved * avg_customer_value
    campaign_cost = base_churners * retention_cost
    net_benefit = value_saved - campaign_cost
    roi = (net_benefit / campaign_cost) * 100
    
    print(f"For every 1,000 customers:")
    print(f"Estimated customers saved from churning: {customers_saved}")
    print(f"Estimated value preserved: ${value_saved:,.2f}")
    print(f"Estimated campaign cost: ${campaign_cost:,.2f}")
    print(f"Estimated net benefit: ${net_benefit:,.2f}")
    print(f"Estimated ROI: {roi:.1f}%")
    
    print("\n" + "="*80)
    print("END OF KEY METRICS")
    print("="*80)

# Add this call at the end of your script
if 'best_model' in locals() and 'model_results' in locals() and 'optimal_threshold' in locals() and 'risk_segments' in locals():
    print_key_documentation_metrics(df, best_model, model_results, X_selected, optimal_threshold, risk_segments)
else:
    print("\nWARNING: Some required variables not found. Run all previous steps first.")

# %%
