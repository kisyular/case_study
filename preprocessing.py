"""
Main Preprocessing Pipeline - Team Integration
Combines work from all three team members: Tina, Andrew, and Rellikson

This script orchestrates the complete preprocessing pipeline by:
1. Running Tina's data inspection and basic EDA
2. Running Andrew's advanced EDA and basic preprocessing  
3. Running Rellikson's feature engineering and transformations
4. Producing final model-ready dataset

Usage:
    python preprocessing.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import team member classes
from tina_preprocessing import TinaPreprocessor
from andrew_preprocessing import AndrewPreprocessor  
from rellikson_preprocessing import RelliksonPreprocessor

class TeamPreprocessingPipeline:
    """
    Main preprocessing pipeline that coordinates all team members' work
    Following the systematic approach from Finalist_2020_4.pdf
    """
    
    def __init__(self, data_path='Case_Data.csv'):
        self.data_path = data_path
        self.df = None
        self.preprocessing_log = []
        
    def log_step(self, step, details):
        """Log preprocessing steps for documentation"""
        self.preprocessing_log.append({
            'step': step,
            'details': details,
            'shape': self.df.shape if self.df is not None else None
        })
        print(f"✓ {step}: {details}")
        
    def run_pipeline(self):
        """
        Execute the complete preprocessing pipeline
        """
        print("=" * 60)
        print("TEAM PREPROCESSING PIPELINE")
        print("Following Finalist_2020_4.pdf Methodology")
        print("=" * 60)
        
        # Step 1: Tina's Work - Data Inspection & Basic EDA
        print("\n1. TINA'S PHASE: Data Inspection & Basic EDA")
        print("-" * 50)
        
        tina = TinaPreprocessor(self.data_path)
        tina.run_full_analysis()
        self.df = tina.get_processed_data()
        self.log_step("Tina's Analysis", f"Basic EDA completed, dataset shape: {self.df.shape}")
        
        # Step 2: Andrew's Work - Advanced EDA & Basic Preprocessing
        print("\n2. ANDREW'S PHASE: Advanced EDA & Basic Preprocessing")
        print("-" * 50)
        
        andrew = AndrewPreprocessor(dataframe=self.df)
        andrew.run_full_analysis() 
        self.df = andrew.get_processed_data()
        self.log_step("Andrew's Analysis", f"Advanced preprocessing completed, dataset shape: {self.df.shape}")
        
        # Step 3: Rellikson's Work - Feature Engineering & Transformations
        print("\n3. RELLIKSON'S PHASE: Feature Engineering & Transformations")
        print("-" * 50)
        
        rellikson = RelliksonPreprocessor(dataframe=self.df)
        rellikson.run_full_analysis()
        self.df = rellikson.get_processed_data()
        self.log_step("Rellikson's Analysis", f"Feature engineering completed, dataset shape: {self.df.shape}")
        
        # Step 4: Final Integration & Validation
        print("\n4. FINAL INTEGRATION & VALIDATION")
        print("-" * 50)
        
        self._final_validation()
        self._save_final_outputs()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        self._print_summary()
        
    def _final_validation(self):
        """Final validation checks on the processed dataset"""
        
        # Check for missing values
        missing_check = self.df.isnull().sum().sum()
        self.log_step("Missing Values Check", f"Total missing values: {missing_check}")
        
        # Check data types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.log_step("Data Types Check", f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}")
        
        # Check target variable
        if 'TARGET' in self.df.columns:
            target_distribution = self.df['TARGET'].value_counts()
            self.log_step("Target Check", f"Class distribution: {target_distribution.to_dict()}")
        
        # Check for infinite values
        inf_check = np.isinf(self.df.select_dtypes(include=[np.number])).sum().sum()
        self.log_step("Infinite Values Check", f"Total infinite values: {inf_check}")
        
        # Memory usage
        memory_usage = self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        self.log_step("Memory Usage", f"{memory_usage:.2f} MB")
        
    def _save_final_outputs(self):
        """Save all final outputs from the preprocessing pipeline"""
        
        # Save final processed dataset
        final_path = 'team_processed_dataset.csv'
        self.df.to_csv(final_path, index=False)
        self.log_step("Final Dataset Saved", f"Saved to {final_path}")
        
        # Save preprocessing log
        log_df = pd.DataFrame(self.preprocessing_log)
        log_df.to_csv('team_preprocessing_log.csv', index=False)
        self.log_step("Preprocessing Log Saved", "Saved to team_preprocessing_log.csv")
        
        # Save feature summary
        self._save_feature_summary()
        
    def _save_feature_summary(self):
        """Create and save a comprehensive feature summary"""
        
        feature_summary = []
        
        for col in self.df.columns:
            if col == 'TARGET':
                feature_type = 'Target Variable'
                source = 'Original'
            elif col.startswith(('Financial_Stress', 'Credit_Stability', 'Property_Burden')):
                feature_type = 'Composite Metric'  
                source = 'Rellikson (Feature Engineering)'
            elif col.startswith(('Customer_Segment', 'Anomaly_Score')):
                feature_type = 'ML-derived Feature'
                source = 'Rellikson (ML Features)'
            elif col.endswith(('_encoded', '_interaction', '_ratio')):
                feature_type = 'Engineered Feature'
                source = 'Andrew/Rellikson (Transformations)'
            elif col.startswith('AGE_') or col.endswith('_GROUP'):
                feature_type = 'Derived Feature'
                source = 'Andrew (Basic Preprocessing)'
            else:
                feature_type = 'Original Feature'
                source = 'Tina (Original Dataset)'
                
            feature_summary.append({
                'Feature': col,
                'Type': feature_type,
                'Source': source,
                'Data_Type': str(self.df[col].dtype),
                'Missing_Count': self.df[col].isnull().sum(),
                'Unique_Values': self.df[col].nunique()
            })
        
        summary_df = pd.DataFrame(feature_summary)
        summary_df.to_csv('team_feature_summary.csv', index=False)
        self.log_step("Feature Summary Saved", "Saved to team_feature_summary.csv")
        
    def _print_summary(self):
        """Print final pipeline summary"""
        print(f"\nFINAL DATASET SUMMARY:")
        print(f"• Shape: {self.df.shape}")
        print(f"• Features: {self.df.shape[1]} columns")
        print(f"• Samples: {self.df.shape[0]} rows")
        
        if 'TARGET' in self.df.columns:
            default_rate = (self.df['TARGET'].sum() / len(self.df)) * 100
            print(f"• Default Rate: {default_rate:.1f}%")
            
        print(f"\nFILES CREATED:")
        print(f"• team_processed_dataset.csv - Final model-ready dataset")
        print(f"• team_preprocessing_log.csv - Complete processing log")
        print(f"• team_feature_summary.csv - Feature documentation")
        print(f"• Individual outputs from each team member in their respective folders")
        
        print(f"\nTEAM CONTRIBUTIONS:")
        print(f"• Tina: Data inspection, basic EDA, quality assessment")
        print(f"• Andrew: Advanced EDA, categorical encoding, basic transformations") 
        print(f"• Rellikson: Feature engineering, composite metrics, ML features")
        
        print(f"\nMETHODOLOGY:")
        print(f"• Based on prize-winning approach from Finalist_2020_4.pdf")
        print(f"• Systematic feature engineering with composite indices")
        print(f"• Customer segmentation and anomaly detection")
        print(f"• Ready for model training and validation")

def main():
    """Main function to run the complete preprocessing pipeline"""
    pipeline = TeamPreprocessingPipeline('Case_Data.csv')
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()