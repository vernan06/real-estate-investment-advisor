"""
Real Estate Investment Advisor - Complete Preprocessing Pipeline
================================================================

This module contains all preprocessing, feature engineering, and transformation
functions needed to transform raw user input into model-ready features.

Pipeline Steps:
1. Data Loading & Initial Cleaning
2. Log Transformation (Price_per_SqFt)
3. Winsorization (Outlier Treatment)
4. Frequency Encoding (City, Locality)
5. Ordinal Encoding (Furnished_Status, Public_Transport_Accessibility)
6. One-Hot Encoding (State, Property_Type, Parking_Space, Security, Facing, Owner_Type, Availability_Status)
7. Data Scaling (Standard, MinMax, Robust)
8. Target Variable Creation (Good_Investment, Future_Price_5Y)
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import (
    OrdinalEncoder, OneHotEncoder, 
    StandardScaler, MinMaxScaler, RobustScaler
)
from sklearn.compose import ColumnTransformer
import joblib
import os

# ============================================================================
# CONFIGURATION - Feature Lists
# ============================================================================

# Original dataset columns (23 features)
ORIGINAL_COLUMNS = [
    'ID', 'State', 'City', 'Locality', 'Property_Type', 'BHK', 'Size_in_SqFt',
    'Price_in_Lakhs', 'Price_per_SqFt', 'Year_Built', 'Furnished_Status',
    'Floor_No', 'Total_Floors', 'Age_of_Property', 'Nearby_Schools',
    'Nearby_Hospitals', 'Public_Transport_Accessibility', 'Parking_Space',
    'Security', 'Amenities', 'Facing', 'Owner_Type', 'Availability_Status'
]

# Columns to ONE-HOT ENCODE
ONEHOT_COLS = [
    'State', 'Property_Type', 'Parking_Space', 'Security',
    'Facing', 'Owner_Type', 'Availability_Status'
]

# Columns to ORDINAL ENCODE
ORDINAL_COLS = ['Furnished_Status', 'Public_Transport_Accessibility']

# Ordinal encoding mappings (order matters!)
ORDINAL_MAPPINGS = {
    'Furnished_Status': ['Unfurnished', 'Semi-furnished', 'Furnished'],
    'Public_Transport_Accessibility': ['Low', 'Medium', 'High']
}

# Columns for FREQUENCY ENCODING
FREQUENCY_COLS = ['City', 'Locality']

# Columns for SCALING
STANDARD_SCALE_COLS = ['Price_in_Lakhs', 'Size_in_SqFt', 'Year_Built']
MINMAX_SCALE_COLS = ['Price_per_SqFt_log', 'BHK']
ROBUST_SCALE_COLS = ['Floor_No', 'Total_Floors', 'Age_of_Property']

# Numerical columns for model
NUMERICAL_COLS = [
    'BHK', 'Size_in_SqFt', 'Price_in_Lakhs', 'Price_per_SqFt',
    'Year_Built', 'Floor_No', 'Total_Floors', 'Age_of_Property',
    'Nearby_Schools', 'Nearby_Hospitals', 'Furnished_Status',
    'Public_Transport_Accessibility', 'City_freq', 'Locality_freq',
    'Price_per_SqFt_log', 'Price_per_SqFt_winsor'
]


# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def load_frequency_mappings(df: pd.DataFrame) -> dict:
    """
    Create frequency encoding mappings from training data.
    Must be called during training and saved for inference.
    
    Args:
        df: Training DataFrame
        
    Returns:
        dict: {column_name: {value: frequency}}
    """
    freq_maps = {}
    for col in FREQUENCY_COLS:
        freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
    return freq_maps


def apply_log_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log transformation to reduce skewness in Price_per_SqFt.
    Uses log1p to handle zero values safely.
    
    Args:
        df: DataFrame with Price_per_SqFt column
        
    Returns:
        DataFrame with new Price_per_SqFt_log column
    """
    df = df.copy()
    df['Price_per_SqFt_log'] = np.log1p(df['Price_per_SqFt'])
    return df


def apply_winsorization(df: pd.DataFrame, limits=(0.01, 0.01)) -> pd.DataFrame:
    """
    Apply Winsorization to cap outliers at 1st and 99th percentile.
    
    Args:
        df: DataFrame with Price_per_SqFt column
        limits: Tuple of (lower_limit, upper_limit) proportions
        
    Returns:
        DataFrame with new Price_per_SqFt_winsor column
    """
    df = df.copy()
    df['Price_per_SqFt_winsor'] = winsorize(df['Price_per_SqFt'], limits=limits)
    return df


def apply_frequency_encoding(df: pd.DataFrame, freq_maps: dict = None) -> pd.DataFrame:
    """
    Apply frequency encoding to high-cardinality columns (City, Locality).
    
    Args:
        df: DataFrame with City and Locality columns
        freq_maps: Pre-computed frequency mappings (for inference)
                   If None, creates new mappings from df (for training)
        
    Returns:
        DataFrame with City_freq and Locality_freq columns
    """
    df = df.copy()
    
    if freq_maps is None:
        freq_maps = load_frequency_mappings(df)
    
    for col in FREQUENCY_COLS:
        # Handle unseen values with a small default frequency
        default_freq = 1e-6
        df[col + '_freq'] = df[col].map(freq_maps.get(col, {})).fillna(default_freq)
    
    return df


def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply ordinal encoding to ordered categorical variables.
    
    Mappings:
    - Furnished_Status: Unfurnished(0) < Semi-furnished(1) < Furnished(2)
    - Public_Transport_Accessibility: Low(0) < Medium(1) < High(2)
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with ordinal encoded columns
    """
    df = df.copy()
    
    # Manual mapping for safety and interpretability
    furnished_map = {'Unfurnished': 0, 'Semi-furnished': 1, 'Furnished': 2}
    transport_map = {'Low': 0, 'Medium': 1, 'High': 2}
    
    df['Furnished_Status'] = df['Furnished_Status'].map(furnished_map)
    df['Public_Transport_Accessibility'] = df['Public_Transport_Accessibility'].map(transport_map)
    
    return df


def apply_onehot_encoding(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    Apply one-hot encoding to categorical variables.
    
    Columns: State, Property_Type, Parking_Space, Security, 
             Facing, Owner_Type, Availability_Status
    
    Args:
        df: DataFrame with categorical columns
        drop_first: Whether to drop first category to avoid multicollinearity
        
    Returns:
        DataFrame with one-hot encoded columns
    """
    df = df.copy()
    df = pd.get_dummies(df, columns=ONEHOT_COLS, drop_first=drop_first)
    return df


def create_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create Good_Investment target variable for classification.
    
    A property is a Good Investment if ANY of these conditions are met:
    1. Price is below median Price_in_Lakhs
    2. Price per SqFt is below median Price_per_SqFt
    3. Investment_Score >= 2
    
    Investment_Score components:
    - BHK >= 3 (+1)
    - Furnished_Status in ['Semi-furnished', 'Furnished'] (+1)
    - Availability_Status is NOT Under_Construction (+1)
    
    Args:
        df: DataFrame with required columns
        
    Returns:
        DataFrame with Good_Investment column (0 or 1)
    """
    df = df.copy()
    
    # Compute medians
    median_price = df['Price_in_Lakhs'].median()
    median_ppsf = df['Price_per_SqFt'].median()
    
    # Compute Investment Score
    # Note: After ordinal encoding, Furnished_Status is numeric (0, 1, 2)
    # Semi-furnished=1, Furnished=2
    df['Investment_Score'] = (
        (df['BHK'] >= 3).astype(int) +
        (df['Furnished_Status'] >= 1).astype(int) +  # Semi-furnished or Furnished
        (df.get('Availability_Status_Under_Construction', 0) == 0).astype(int)
    )
    
    # Final classification target
    df['Good_Investment'] = (
        (df['Price_in_Lakhs'] <= median_price) |
        (df['Price_per_SqFt'] <= median_ppsf) |
        (df['Investment_Score'] >= 2)
    ).astype(int)
    
    return df


def create_regression_target(df: pd.DataFrame, rate: float = 0.08, years: int = 5) -> pd.DataFrame:
    """
    Create Future_Price_5Y target variable for regression.
    
    Formula: Future_Price = Current_Price * (1 + rate)^years
    
    Args:
        df: DataFrame with Price_in_Lakhs column
        rate: Annual appreciation rate (default 8%)
        years: Number of years to project (default 5)
        
    Returns:
        DataFrame with Future_Price_5Y column
    """
    df = df.copy()
    df['Future_Price_5Y'] = df['Price_in_Lakhs'] * ((1 + rate) ** years)
    return df


def create_preprocessing_pipeline(numeric_cols: list, categorical_cols: list):
    """
    Create sklearn ColumnTransformer for preprocessing.
    Used in model Pipeline for automated preprocessing.
    
    Args:
        numeric_cols: List of numerical column names
        categorical_cols: List of categorical column names
        
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("scale", StandardScaler(), numeric_cols)
        ]
    )
    return preprocessor


def preprocess_raw_input(input_data: dict, freq_maps: dict) -> pd.DataFrame:
    """
    MAIN FUNCTION: Preprocess raw user input for model prediction.
    
    This function takes raw user input and applies ALL preprocessing steps
    to create model-ready features.
    
    Args:
        input_data: Dictionary with raw property features
        freq_maps: Pre-computed frequency mappings for City/Locality
        
    Returns:
        DataFrame: Preprocessed features ready for model prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Step 1: Log Transformation
    df = apply_log_transformation(df)
    
    # Step 2: Winsorization
    df = apply_winsorization(df)
    
    # Step 3: Frequency Encoding
    df = apply_frequency_encoding(df, freq_maps)
    
    # Step 4: Ordinal Encoding
    df = apply_ordinal_encoding(df)
    
    # Step 5: One-Hot Encoding
    df = apply_onehot_encoding(df)
    
    return df


def full_preprocessing_pipeline(df: pd.DataFrame, is_training: bool = True, 
                                freq_maps: dict = None) -> tuple:
    """
    Complete preprocessing pipeline from raw data to model-ready features.
    
    Args:
        df: Raw DataFrame
        is_training: If True, creates new freq_maps. If False, uses provided maps.
        freq_maps: Pre-computed frequency mappings (required if is_training=False)
        
    Returns:
        tuple: (processed_df, freq_maps)
    """
    df = df.copy()
    
    # Step 1: Log Transformation
    df = apply_log_transformation(df)
    print("✓ Log Transformation completed")
    
    # Step 2: Winsorization
    df = apply_winsorization(df)
    print("✓ Winsorization completed")
    
    # Step 3: Frequency Encoding
    if is_training:
        freq_maps = load_frequency_mappings(df)
    df = apply_frequency_encoding(df, freq_maps)
    print("✓ Frequency Encoding completed")
    
    # Step 4: Ordinal Encoding
    df = apply_ordinal_encoding(df)
    print("✓ Ordinal Encoding completed")
    
    # Step 5: One-Hot Encoding
    df = apply_onehot_encoding(df)
    print("✓ One-Hot Encoding completed")
    
    return df, freq_maps


def save_preprocessing_artifacts(freq_maps: dict, save_dir: str = 'models'):
    """
    Save preprocessing artifacts for deployment.
    
    Args:
        freq_maps: Frequency encoding mappings
        save_dir: Directory to save artifacts
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save frequency maps
    joblib.dump(freq_maps, os.path.join(save_dir, 'freq_maps.pkl'))
    print(f"✓ Saved frequency maps to {save_dir}/freq_maps.pkl")


def load_preprocessing_artifacts(save_dir: str = 'models') -> dict:
    """
    Load preprocessing artifacts for inference.
    
    Args:
        save_dir: Directory containing artifacts
        
    Returns:
        dict: Frequency encoding mappings
    """
    freq_maps = joblib.load(os.path.join(save_dir, 'freq_maps.pkl'))
    print(f"✓ Loaded frequency maps from {save_dir}/freq_maps.pkl")
    return freq_maps


# ============================================================================
# FEATURE METADATA (for Streamlit app)
# ============================================================================

FEATURE_METADATA = {
    'State': {
        'type': 'categorical',
        'description': 'State where property is located',
        'values': None  # Will be populated from data
    },
    'City': {
        'type': 'categorical',
        'description': 'City where property is located',
        'values': None
    },
    'Locality': {
        'type': 'categorical', 
        'description': 'Specific locality/neighborhood',
        'values': None
    },
    'Property_Type': {
        'type': 'categorical',
        'description': 'Type of property',
        'values': ['Apartment', 'Villa', 'Independent House', 'Builder Floor', 'Studio']
    },
    'BHK': {
        'type': 'numeric',
        'description': 'Number of bedrooms',
        'min': 1,
        'max': 6,
        'step': 1
    },
    'Size_in_SqFt': {
        'type': 'numeric',
        'description': 'Property size in square feet',
        'min': 100,
        'max': 10000,
        'step': 50
    },
    'Price_in_Lakhs': {
        'type': 'numeric',
        'description': 'Property price in Lakhs INR',
        'min': 1,
        'max': 1000,
        'step': 1
    },
    'Price_per_SqFt': {
        'type': 'numeric',
        'description': 'Price per square foot (auto-calculated)',
        'min': 0,
        'max': 50000,
        'calculated': True
    },
    'Year_Built': {
        'type': 'numeric',
        'description': 'Year property was built',
        'min': 1950,
        'max': 2025,
        'step': 1
    },
    'Furnished_Status': {
        'type': 'categorical',
        'description': 'Furnishing level',
        'values': ['Unfurnished', 'Semi-furnished', 'Furnished']
    },
    'Floor_No': {
        'type': 'numeric',
        'description': 'Floor number',
        'min': 0,
        'max': 50,
        'step': 1
    },
    'Total_Floors': {
        'type': 'numeric',
        'description': 'Total floors in building',
        'min': 1,
        'max': 60,
        'step': 1
    },
    'Age_of_Property': {
        'type': 'numeric',
        'description': 'Age of property in years (auto-calculated)',
        'min': 0,
        'max': 100,
        'calculated': True
    },
    'Nearby_Schools': {
        'type': 'numeric',
        'description': 'Number of nearby schools',
        'min': 0,
        'max': 10,
        'step': 1
    },
    'Nearby_Hospitals': {
        'type': 'numeric',
        'description': 'Number of nearby hospitals',
        'min': 0,
        'max': 10,
        'step': 1
    },
    'Public_Transport_Accessibility': {
        'type': 'categorical',
        'description': 'Quality of public transport access',
        'values': ['Low', 'Medium', 'High']
    },
    'Parking_Space': {
        'type': 'categorical',
        'description': 'Parking availability',
        'values': ['None', 'Open', 'Covered']
    },
    'Security': {
        'type': 'categorical',
        'description': 'Security features',
        'values': ['None', 'Security Guard', 'CCTV', 'Gated Community']
    },
    'Amenities': {
        'type': 'text',
        'description': 'Available amenities (comma-separated)',
        'values': ['Gym', 'Pool', 'Clubhouse', 'Garden', 'Playground']
    },
    'Facing': {
        'type': 'categorical',
        'description': 'Direction property faces',
        'values': ['North', 'South', 'East', 'West', 'North-East', 'North-West', 'South-East', 'South-West']
    },
    'Owner_Type': {
        'type': 'categorical',
        'description': 'Type of property owner',
        'values': ['Individual', 'Builder', 'Broker/Agent']
    },
    'Availability_Status': {
        'type': 'categorical',
        'description': 'Current availability status',
        'values': ['Available', 'Sold', 'Under Construction']
    }
}


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Real Estate Investment Advisor - Preprocessing Pipeline")
    print("=" * 60)
    
    # Example raw input
    test_input = {
        'State': 'Maharashtra',
        'City': 'Mumbai',
        'Locality': 'Andheri West',
        'Property_Type': 'Apartment',
        'BHK': 3,
        'Size_in_SqFt': 1500,
        'Price_in_Lakhs': 150,
        'Price_per_SqFt': 10000,
        'Year_Built': 2020,
        'Furnished_Status': 'Semi-furnished',
        'Floor_No': 5,
        'Total_Floors': 15,
        'Age_of_Property': 5,
        'Nearby_Schools': 3,
        'Nearby_Hospitals': 2,
        'Public_Transport_Accessibility': 'High',
        'Parking_Space': 'Covered',
        'Security': 'Gated Community',
        'Amenities': 'Gym, Pool',
        'Facing': 'East',
        'Owner_Type': 'Builder',
        'Availability_Status': 'Available'
    }
    
    print("\nTest Input:")
    for k, v in test_input.items():
        print(f"  {k}: {v}")
    
    print("\nPipeline Test Complete!")
