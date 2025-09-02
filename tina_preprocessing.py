"""
Case Competition: Data Pre-processing for Predictive Modeling
Team Member: TINA
Role: Data Inspector & Basic EDA Specialist
Responsible for: Initial data inspection, basic EDA, data quality assessment

Variables I'm working with:
- Overall dataset structure and dimensions
- Target variable (TARGET) analysis
- Basic demographic variables: CODE_GENDER, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS
- Financial overview variables: AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY
- Missing value patterns across all variables
- Duplicate detection and basic data quality checks

My preprocessing decisions:
1. Focus on understanding data structure and quality issues
2. Identify patterns in missing values
3. Basic descriptive statistics for all variables
4. Initial visualizations for key variables
5. Document data quality issues for team discussion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TinaDataInspector:
    """
    Tina's responsibility: Basic EDA and Data Quality Assessment
    """
    
    def __init__(self):
        self.df = None
        self.data_quality_report = {}
        
    def load_and_inspect_data(self, filepath='Case_Data.csv'):
        """Load data and perform initial inspection"""
        print("=" * 60)
        print("TINA'S SECTION: BASIC DATA INSPECTION & EDA")
        print("=" * 60)
        
        # Load data
        self.df = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]:,} columns")
        print(f"💾 Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return self.df
    
    def basic_data_overview(self):
        """Comprehensive basic data overview"""
        print(f"\n🔍 BASIC DATA OVERVIEW:")
        print("-" * 40)
        
        # Data types summary
        dtype_summary = self.df.dtypes.value_counts()
        print(f"📈 Data Types Distribution:")
        for dtype, count in dtype_summary.items():
            print(f"   - {dtype}: {count} columns")
        
        # Basic statistics for numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print(f"\n📊 Variable Types:")
        print(f"   - Numerical variables: {len(numeric_cols)}")
        print(f"   - Categorical variables: {len(categorical_cols)}")
        
        # Save basic info to report
        self.data_quality_report['shape'] = self.df.shape
        self.data_quality_report['numeric_cols'] = len(numeric_cols)
        self.data_quality_report['categorical_cols'] = len(categorical_cols)
        
        return self.df.info()
    
    def target_variable_analysis(self):
        """Detailed analysis of TARGET variable - critical for loan default prediction"""
        print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
        print("-" * 40)
        
        if 'TARGET' not in self.df.columns:
            print("❌ TARGET variable not found!")
            return
        
        # Basic TARGET statistics
        target_stats = self.df['TARGET'].describe()
        target_counts = self.df['TARGET'].value_counts()
        target_props = self.df['TARGET'].value_counts(normalize=True) * 100
        
        print(f"📊 Target Variable Distribution:")
        print(f"   - No Default (0): {target_counts[0]:,} customers ({target_props[0]:.1f}%)")
        print(f"   - Default (1): {target_counts[1]:,} customers ({target_props[1]:.1f}%)")
        
        # Calculate class imbalance ratio
        imbalance_ratio = target_counts[0] / target_counts[1] if target_counts[1] > 0 else float('inf')
        print(f"   - Class Imbalance Ratio: {imbalance_ratio:.1f}:1")
        
        # Critical insight for modeling
        if imbalance_ratio > 10:
            print(f"   ⚠️  SEVERE CLASS IMBALANCE DETECTED!")
            print(f"       This will require special handling in modeling (SMOTE, class weights, etc.)")
        
        # Save to report
        self.data_quality_report['target_imbalance'] = imbalance_ratio
        self.data_quality_report['default_rate'] = target_props[1]
        
        # Visualization
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        target_counts.plot(kind='bar', color=['lightblue', 'salmon'])
        plt.title('Target Distribution (Count)')
        plt.xlabel('TARGET (0=No Default, 1=Default)')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        target_props.plot(kind='bar', color=['lightblue', 'salmon'])
        plt.title('Target Distribution (%)')
        plt.xlabel('TARGET (0=No Default, 1=Default)')
        plt.ylabel('Percentage')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('tina_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return target_stats
    
    def demographic_variables_analysis(self):
        """Analysis of key demographic variables"""
        print(f"\n👥 DEMOGRAPHIC VARIABLES ANALYSIS:")
        print("-" * 40)
        
        demographic_vars = ['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE']
        
        for var in demographic_vars:
            if var in self.df.columns:
                print(f"\n📋 {var}:")
                unique_count = self.df[var].nunique()
                print(f"   - Unique values: {unique_count}")
                
                # Show distribution
                value_counts = self.df[var].value_counts()
                print(f"   - Distribution:")
                for value, count in value_counts.head(5).items():
                    pct = (count / len(self.df)) * 100
                    print(f"     • {value}: {count:,} ({pct:.1f}%)")
                
                if unique_count > 5:
                    print(f"     ... and {unique_count - 5} more categories")
                
                # Check for relationship with target
                if 'TARGET' in self.df.columns:
                    target_by_category = self.df.groupby(var)['TARGET'].mean()
                    max_default_rate = target_by_category.max()
                    min_default_rate = target_by_category.min()
                    print(f"   - Default rate range: {min_default_rate:.1%} to {max_default_rate:.1%}")
                    
                    if (max_default_rate - min_default_rate) > 0.05:  # 5% difference
                        print(f"     ✨ SIGNIFICANT predictor - consider for modeling!")
            else:
                print(f"   ❌ {var} not found in dataset")
    
    def financial_variables_overview(self):
        """Overview of key financial variables"""
        print(f"\n💰 FINANCIAL VARIABLES OVERVIEW:")
        print("-" * 40)
        
        financial_vars = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
        
        financial_summary = {}
        
        for var in financial_vars:
            if var in self.df.columns:
                print(f"\n💵 {var}:")
                stats = self.df[var].describe()
                
                print(f"   - Mean: ${stats['mean']:,.0f}")
                print(f"   - Median: ${stats['50%']:,.0f}")
                print(f"   - Min: ${stats['min']:,.0f}")
                print(f"   - Max: ${stats['max']:,.0f}")
                
                # Check for zeros
                zero_count = (self.df[var] == 0).sum()
                zero_pct = (zero_count / len(self.df)) * 100
                if zero_count > 0:
                    print(f"   - Zero values: {zero_count:,} ({zero_pct:.1f}%)")
                
                # Check for extreme outliers
                q1 = stats['25%']
                q3 = stats['75%']
                iqr = q3 - q1
                outlier_threshold_high = q3 + 3 * iqr
                outliers = (self.df[var] > outlier_threshold_high).sum()
                if outliers > 0:
                    outlier_pct = (outliers / len(self.df)) * 100
                    print(f"   - Extreme outliers: {outliers:,} ({outlier_pct:.1f}%)")
                
                financial_summary[var] = {
                    'mean': stats['mean'],
                    'median': stats['50%'],
                    'zero_count': zero_count,
                    'outliers': outliers
                }
            else:
                print(f"   ❌ {var} not found in dataset")
        
        # Create financial overview visualization
        self._create_financial_visualizations()
        
        return financial_summary
    
    def _create_financial_visualizations(self):
        """Create visualizations for financial variables"""
        financial_vars = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
        existing_vars = [var for var in financial_vars if var in self.df.columns]
        
        if len(existing_vars) >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, var in enumerate(existing_vars[:4]):
                if i < len(axes):
                    # Use log scale for better visualization of skewed financial data
                    data = self.df[var].dropna()
                    if (data > 0).all():
                        axes[i].hist(np.log10(data), bins=30, alpha=0.7, color='skyblue')
                        axes[i].set_title(f'Log10({var}) Distribution')
                        axes[i].set_xlabel(f'Log10({var})')
                    else:
                        axes[i].hist(data, bins=30, alpha=0.7, color='skyblue')
                        axes[i].set_title(f'{var} Distribution')
                        axes[i].set_xlabel(var)
                    
                    axes[i].set_ylabel('Frequency')
            
            # Remove empty subplots
            for i in range(len(existing_vars), len(axes)):
                axes[i].remove()
            
            plt.tight_layout()
            plt.savefig('tina_financial_distributions.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def missing_values_analysis(self):
        """Comprehensive missing values analysis"""
        print(f"\n🔍 MISSING VALUES ANALYSIS:")
        print("-" * 40)
        
        # Calculate missing values
        missing_counts = self.df.isnull().sum()
        missing_percent = (missing_counts / len(self.df)) * 100
        
        # Create missing values summary
        missing_summary = pd.DataFrame({
            'Variable': missing_counts.index,
            'Missing_Count': missing_counts.values,
            'Missing_Percent': missing_percent.values
        }).sort_values('Missing_Percent', ascending=False)
        
        # Filter only variables with missing values
        missing_summary = missing_summary[missing_summary.Missing_Count > 0]
        
        if len(missing_summary) == 0:
            print("✅ No missing values detected!")
            return None
        
        print(f"📊 Found {len(missing_summary)} variables with missing values:")
        
        # Categorize missing patterns
        high_missing = missing_summary[missing_summary.Missing_Percent > 30]
        medium_missing = missing_summary[(missing_summary.Missing_Percent >= 10) & 
                                       (missing_summary.Missing_Percent <= 30)]
        low_missing = missing_summary[(missing_summary.Missing_Percent > 0) & 
                                    (missing_summary.Missing_Percent < 10)]
        
        print(f"\n🚨 HIGH missing (>30%): {len(high_missing)} variables")
        for _, row in high_missing.head(5).iterrows():
            print(f"   - {row['Variable']}: {row['Missing_Percent']:.1f}%")
        
        print(f"\n⚠️  MEDIUM missing (10-30%): {len(medium_missing)} variables")
        for _, row in medium_missing.head(5).iterrows():
            print(f"   - {row['Variable']}: {row['Missing_Percent']:.1f}%")
        
        print(f"\n✋ LOW missing (<10%): {len(low_missing)} variables")
        for _, row in low_missing.head(10).iterrows():
            print(f"   - {row['Variable']}: {row['Missing_Percent']:.1f}%")
        
        # Save to report
        self.data_quality_report['missing_high'] = len(high_missing)
        self.data_quality_report['missing_medium'] = len(medium_missing)
        self.data_quality_report['missing_low'] = len(low_missing)
        
        # Create missing values visualization
        if len(missing_summary) > 0:
            plt.figure(figsize=(12, 8))
            top_missing = missing_summary.head(20)
            
            plt.barh(range(len(top_missing)), top_missing['Missing_Percent'])
            plt.yticks(range(len(top_missing)), top_missing['Variable'])
            plt.xlabel('Percentage Missing (%)')
            plt.title('Top 20 Variables by Missing Value Percentage')
            plt.gca().invert_yaxis()
            
            # Add color coding
            colors = ['red' if x > 30 else 'orange' if x > 10 else 'yellow' 
                     for x in top_missing['Missing_Percent']]
            bars = plt.barh(range(len(top_missing)), top_missing['Missing_Percent'], color=colors)
            
            plt.tight_layout()
            plt.savefig('tina_missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return missing_summary
    
    def duplicate_detection(self):
        """Check for duplicate rows"""
        print(f"\n🔄 DUPLICATE DETECTION:")
        print("-" * 40)
        
        # Check for exact duplicates
        duplicates = self.df.duplicated()
        duplicate_count = duplicates.sum()
        
        print(f"📊 Duplicate Analysis:")
        print(f"   - Exact duplicate rows: {duplicate_count:,}")
        
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / len(self.df)) * 100
            print(f"   - Percentage of dataset: {duplicate_pct:.2f}%")
            print(f"   ⚠️  ACTION REQUIRED: Consider removing duplicates")
            
            # Show some examples
            duplicate_rows = self.df[duplicates]
            print(f"\n📋 Example duplicate rows (first 3):")
            print(duplicate_rows.head(3)[['TARGET', 'CODE_GENDER', 'AMT_INCOME_TOTAL']].to_string())
        else:
            print(f"   ✅ No duplicate rows found!")
        
        # Check for potential duplicates based on key variables
        if duplicate_count == 0:
            key_vars = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']
            available_vars = [var for var in key_vars if var in self.df.columns]
            
            if len(available_vars) >= 2:
                potential_duplicates = self.df.duplicated(subset=available_vars)
                potential_count = potential_duplicates.sum()
                
                if potential_count > 0:
                    print(f"   🔍 Potential duplicates (same financial profile): {potential_count:,}")
                    print(f"   💡 These might be legitimate (family members, etc.)")
        
        self.data_quality_report['duplicates'] = duplicate_count
        return duplicate_count
    
    def create_data_quality_summary(self):
        """Create comprehensive data quality summary for the team"""
        print(f"\n📋 DATA QUALITY SUMMARY FOR TEAM:")
        print("=" * 50)
        
        print(f"🏗️  DATASET STRUCTURE:")
        print(f"   - Rows: {self.data_quality_report.get('shape', (0, 0))[0]:,}")
        print(f"   - Columns: {self.data_quality_report.get('shape', (0, 0))[1]:,}")
        print(f"   - Numeric variables: {self.data_quality_report.get('numeric_cols', 0)}")
        print(f"   - Categorical variables: {self.data_quality_report.get('categorical_cols', 0)}")
        
        print(f"\n🎯 TARGET VARIABLE:")
        print(f"   - Default rate: {self.data_quality_report.get('default_rate', 0):.1f}%")
        print(f"   - Class imbalance: {self.data_quality_report.get('target_imbalance', 0):.1f}:1")
        
        print(f"\n🔍 MISSING VALUES:")
        print(f"   - High missing (>30%): {self.data_quality_report.get('missing_high', 0)} variables")
        print(f"   - Medium missing (10-30%): {self.data_quality_report.get('missing_medium', 0)} variables")
        print(f"   - Low missing (<10%): {self.data_quality_report.get('missing_low', 0)} variables")
        
        print(f"\n🔄 DATA QUALITY:")
        print(f"   - Duplicate rows: {self.data_quality_report.get('duplicates', 0)}")
        
        print(f"\n💌 RECOMMENDATIONS FOR ANDREW & RELLIKSON:")
        print("   📌 Priority actions:")
        
        if self.data_quality_report.get('target_imbalance', 0) > 10:
            print("   - 🚨 Handle severe class imbalance (use SMOTE, class weights)")
        
        if self.data_quality_report.get('missing_high', 0) > 0:
            print("   - 🗑️  Consider dropping high-missing variables")
        
        if self.data_quality_report.get('missing_medium', 0) > 0:
            print("   - 🔧 Use domain-informed imputation for medium-missing")
        
        if self.data_quality_report.get('duplicates', 0) > 0:
            print("   - 🧹 Remove duplicate rows")
        
        print("   - 📊 Focus on financial ratios for feature engineering")
        print("   - 🎨 Use target encoding for high-cardinality categoricals")
        
        return self.data_quality_report
    
    def save_tina_results(self):
        """Save Tina's analysis results for team integration"""
        # Save processed data
        output_file = 'tina_data_inspection.csv'
        self.df.to_csv(output_file, index=False)
        
        # Save quality report
        import json
        with open('tina_quality_report.json', 'w') as f:
            json.dump(self.data_quality_report, f, indent=2)
        
        print(f"\n💾 TINA'S DELIVERABLES SAVED:")
        print(f"   - Data file: {output_file}")
        print(f"   - Quality report: tina_quality_report.json")
        print(f"   - Visualizations: tina_*.png files")
        
        return output_file


def main():
    """Execute Tina's data inspection pipeline"""
    print("🚀 TINA'S DATA INSPECTION & BASIC EDA")
    print("Following systematic data quality assessment methodology")
    print("=" * 60)
    
    # Initialize Tina's inspector
    inspector = TinaDataInspector()
    
    try:
        # Execute Tina's pipeline
        df = inspector.load_and_inspect_data('Case_Data.csv')
        inspector.basic_data_overview()
        inspector.target_variable_analysis()
        inspector.demographic_variables_analysis()
        inspector.financial_variables_overview()
        inspector.missing_values_analysis()
        inspector.duplicate_detection()
        inspector.create_data_quality_summary()
        inspector.save_tina_results()
        
        print("\n🎉 TINA'S SECTION COMPLETE!")
        print("✅ Data inspection finished successfully")
        print("📤 Results ready for Andrew's preprocessing phase")
        
    except Exception as e:
        print(f"\n❌ ERROR in Tina's section: {str(e)}")
        raise


if __name__ == "__main__":
    main()


"""
TINA'S DOCUMENTATION:
====================

MY ROLE: Data Inspector & Basic EDA Specialist

VARIABLES I'M RESPONSIBLE FOR:
- TARGET: Target variable analysis and class imbalance detection
- Demographic: CODE_GENDER, NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_HOUSING_TYPE  
- Financial: AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE
- All variables: Missing value patterns, duplicates, basic quality checks

KEY DECISIONS MADE:
1. Target Variable: Identified severe class imbalance (need special handling)
2. Missing Values: Categorized into High (>30%), Medium (10-30%), Low (<10%)
3. Duplicates: Systematic detection of exact and potential duplicates
4. Financial Variables: Identified extreme outliers and zero values
5. Categorical Variables: Assessed cardinality and predictive potential

RECOMMENDATIONS FOR TEAM:
- Andrew: Focus on imputation strategies for medium-missing variables
- Rellikson: Create financial ratios and handle class imbalance
- All: Use target encoding for high-cardinality categoricals

OUTPUT FILES:
- tina_data_inspection.csv: Inspected dataset
- tina_quality_report.json: Detailed quality metrics
- tina_*.png: Visualization files for team review

This foundational analysis provides the team with essential data quality insights
to make informed preprocessing decisions in subsequent phases.
"""