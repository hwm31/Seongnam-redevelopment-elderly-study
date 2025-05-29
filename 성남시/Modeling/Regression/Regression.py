import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Korean font setup with fallback options
def setup_korean_font():
    """Setup Korean font for matplotlib with multiple fallback options"""
    import platform
    import matplotlib.font_manager as fm
    import os
    
    # Try to find available Korean fonts
    font_candidates = []
    system = platform.system()
    
    if system == 'Windows':
        font_candidates = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei']
    elif system == 'Darwin':  # macOS
        font_candidates = ['AppleGothic', 'Apple SD Gothic Neo', 'Helvetica']
    else:  # Linux and others
        font_candidates = ['DejaVu Sans', 'Liberation Sans']
    
    # Get available fonts
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = 'DejaVu Sans'  # fallback
    
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break
    
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"Using font: {selected_font}")
    return selected_font

setup_korean_font()

class SeongnamOddYearAnalyzer:
    def __init__(self, before_data, after_data):
        self.before_data = before_data
        self.after_data = after_data
        self.results = {}
        
        # Feature information for odd years only (홀수년도 특성)
        self.feature_info = {
            '만나이': {'type': 'continuous', 'range': 'age in years'},
            '지역거주기간': {'type': 'continuous', 'range': 'years of residence'},
            '향후거주의향': {'type': 'likert', 'range': '1=strongly agree ~ 5=strongly disagree', 'reverse': True},
            '정주의식': {'type': 'special_categorical', 'range': '1,3=strong attachment; 2,4=weak attachment', 'mapping': {1: 4, 2: 2, 3: 4, 4: 1}},
            '거주지소속감': {'type': 'likert', 'range': '1=none ~ 4=very strong', 'reverse_2023': True},
            '거주지만족도': {'type': 'likert', 'range': '1=very satisfied ~ 5=very dissatisfied', 'reverse': True},
            '월평균가구소득': {'type': 'ordinal', 'range': '1=<100만원 ~ 8=700만원+'},
            '부채유무': {'type': 'binary', 'range': '1=yes, 2=no'},
            '삶의만족도': {'type': 'likert', 'range': '1=very satisfied ~ 5=very dissatisfied', 'reverse': True}
        }
        
    def prepare_data(self, data, target_col='거주지만족도'):
        """Data preparation for odd years only"""
        # Filter only odd years data
        if 'year' in data.columns:
            odd_years = [2017, 2019, 2021, 2023]
            data = data[data['year'].isin(odd_years)].copy()
        
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in [target_col, 'year'] and col in self.feature_info]
        X = data[feature_cols].copy()
        y = data[target_col].copy()
        
        # Process features with special handling for 2023 거주지소속감
        X_processed = self.process_features(X, data)
        
        return X_processed, y, feature_cols
    
    def process_features(self, X, original_data):
        """Process features with year-specific handling"""
        X_processed = X.copy()
        
        for col in X.columns:
            if col in self.feature_info:
                feature_info = self.feature_info[col]
                feature_type = feature_info['type']
                
                if feature_type == 'likert':
                    # Handle 거주지소속감 special case for 2023
                    if col == '거주지소속감' and 'year' in original_data.columns:
                        # For 2023: reverse the scale (1=매우있다 → 4=매우있다, 4=전혀없다 → 1=전혀없다)
                        if 2023 in original_data['year'].values:
                            # Create year-specific processing
                            processed_col = X[col].copy()
                            
                            # Find 2023 rows
                            if len(original_data) == len(X):  # Same index
                                year_2023_mask = (original_data['year'] == 2023)
                                # Reverse 2023 values: 1→4, 2→3, 3→2, 4→1
                                processed_col[year_2023_mask] = 5 - X[col][year_2023_mask]
                            
                            X_processed[col] = processed_col
                        else:
                            X_processed[col] = X[col]  # Keep as is for other years
                    
                    # Handle other likert scales
                    elif feature_info.get('reverse', False):
                        # Reverse satisfaction scales so higher = better
                        max_val = X[col].max()
                        X_processed[col] = (max_val + 1) - X[col]
                    else:
                        X_processed[col] = X[col]
                
                elif feature_type == 'binary':
                    # Convert to 0/1 (1=yes→1, 2=no→0)
                    X_processed[col] = (X[col] == 1).astype(int)
                
                elif feature_type == 'categorical':
                    # Keep as is but ensure numeric
                    X_processed[col] = pd.to_numeric(X[col], errors='coerce')
                
                elif feature_type == 'special_categorical':
                    # Handle 정주의식 special mapping: 1,3=strong(4), 2,4=weak(1-2)
                    if col == '정주의식':
                        mapping = feature_info.get('mapping', {})
                        X_processed[col] = X[col].map(mapping).fillna(X[col])
                    else:
                        X_processed[col] = pd.to_numeric(X[col], errors='coerce')
                
                elif feature_type == 'ordinal':
                    # Keep ordinal encoding
                    X_processed[col] = pd.to_numeric(X[col], errors='coerce')
                
                elif feature_type == 'continuous':
                    # Ensure numeric
                    X_processed[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values with median
        X_processed = X_processed.fillna(X_processed.median())
        
        return X_processed
    
    def analyze_feature_relationships(self, X, y, period_name, save_dir):
        """Analyze correlation and covariance between features including target variable"""
        print(f"\n=== Feature Relationship Analysis for {period_name} ===")
        
        # Combine features and target for comprehensive correlation analysis
        X_with_target = X.copy()
        X_with_target['거주지만족도'] = y
        
        # Calculate correlation matrix
        correlation_matrix = X_with_target.corr()
        
        # Calculate covariance matrix  
        covariance_matrix = X_with_target.cov()
        
        # Find highly correlated pairs (|correlation| > 0.7)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'Feature1': correlation_matrix.columns[i],
                        'Feature2': correlation_matrix.columns[j], 
                        'Correlation': corr_val
                    })
        
        print(f"High correlation pairs (|r| > 0.7): {len(high_corr_pairs)}")
        
        # Show correlations with target variable
        target_correlations = correlation_matrix['거주지만족도'].drop('거주지만족도').sort_values(key=abs, ascending=False)
        print(f"\nCorrelations with Residence Satisfaction:")
        for feature, corr in target_correlations.items():
            print(f"  {feature}: {corr:.3f}")
        
        # Create English feature names for plotting
        feature_mapping = {
            '만나이': 'Age',
            '지역거주기간': 'Residence_Period',
            '향후거주의향': 'Future_Residence_Intent',
            '정주의식': 'Settlement_Mindset',
            '거주지소속감': 'Place_Attachment',
            '거주지만족도': 'Residence_Satisfaction',
            '월평균가구소득': 'Monthly_Income',
            '부채유무': 'Has_Debt',
            '삶의만족도': 'Life_Satisfaction'
        }
        
        # Rename columns for visualization
        corr_matrix_en = correlation_matrix.copy()
        cov_matrix_en = covariance_matrix.copy()
        
        # Map Korean to English names
        new_cols = [feature_mapping.get(col, col) for col in correlation_matrix.columns]
        corr_matrix_en.columns = new_cols
        corr_matrix_en.index = new_cols
        cov_matrix_en.columns = new_cols
        cov_matrix_en.index = new_cols
        
        # Visualize correlation matrix (full matrix including target)
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(corr_matrix_en, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .8})
        
        plt.title(f'Feature Correlation Matrix with Target - {period_name}', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_matrix_with_target_{period_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize covariance matrix
        plt.figure(figsize=(14, 12))
        sns.heatmap(cov_matrix_en, annot=True, fmt='.2f', 
                   cmap='viridis', square=True, cbar_kws={"shrink": .8})
        
        plt.title(f'Feature Covariance Matrix with Target - {period_name}', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/covariance_matrix_with_target_{period_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        return correlation_matrix, covariance_matrix, high_corr_pairs
    
    def create_features(self, X):
        """Create Original Features and PCA Components"""
        # 1. Original Features (standardized)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 2. PCA Components
        pca = PCA(n_components=0.95)  # Retain 95% variance
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        return X_scaled, X_pca, scaler, pca
    
    def create_regression_equation(self, model, feature_names, model_name="", period_name=""):
        """Create multiple regression equation"""
        if not hasattr(model, 'coef_'):
            return "Cannot generate equation for this model type."
        
        intercept = model.intercept_
        coefficients = model.coef_
        
        # Create English equation
        equation = f"\n=== {period_name} - {model_name} Multiple Regression Equation ===\n"
        equation += "Residence_Satisfaction = "
        
        if intercept >= 0:
            equation += f"{intercept:.4f}"
        else:
            equation += f"({intercept:.4f})"
        
        # Map feature names to English
        feature_mapping = {
            '만나이': 'Age',
            '지역거주기간': 'Residence_Period',
            '향후거주의향': 'Future_Residence_Intent',
            '정주의식': 'Settlement_Mindset',
            '거주지소속감': 'Place_Attachment',
            '월평균가구소득': 'Monthly_Income',
            '부채유무': 'Has_Debt',
            '삶의만족도': 'Life_Satisfaction'
        }
        
        for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
            eng_feature = feature_mapping.get(feature, feature)
            if coef >= 0:
                equation += f" + {coef:.4f} × {eng_feature}"
            else:
                equation += f" - {abs(coef):.4f} × {eng_feature}"
        
        equation += "\n"
        
        # Add coefficient interpretation
        equation += f"\n--- Coefficient Interpretation ---\n"
        coef_importance = [(abs(coef), feature, coef) for coef, feature in zip(coefficients, feature_names)]
        coef_importance.sort(reverse=True)
        
        for i, (abs_coef, feature, coef) in enumerate(coef_importance[:5]):
            eng_feature = feature_mapping.get(feature, feature)
            direction = "increases" if coef > 0 else "decreases"
            equation += f"{i+1}. {eng_feature}: {coef:.4f} (1 unit increase → residence satisfaction {direction})\n"
        
        # Add feature meaning explanation
        equation += f"\n--- Feature Meanings (Odd Years Only) ---\n"
        for feature in feature_names:
            if feature in self.feature_info:
                eng_feature = feature_mapping.get(feature, feature)
                meaning = self.feature_info[feature]['range']
                
                # Special notes for processed features
                if feature == '거주지소속감':
                    meaning += " (Note: 2023 values were reversed during processing)"
                elif feature == '정주의식':
                    meaning += " (Note: 1,3→4 strong; 2,4→1-2 weak)"
                
                equation += f"{eng_feature}: {meaning}\n"
        
        return equation
    
    def run_regression_analysis(self, period_name, data, save_dir="regression_plots"):
        """Run multiple regression analysis for odd years"""
        print(f"\n=== {period_name} Multiple Regression Analysis (Odd Years Only) ===")
        
        # Data preparation
        X, y, feature_cols = self.prepare_data(data)
        
        print(f"Odd years features processed:")
        for col in feature_cols:
            if col in self.feature_info:
                meaning = self.feature_info[col]['range']
                if col == '거주지소속감':
                    meaning += " (2023 reversed)"
                elif col == '정주의식':
                    meaning += " (1,3→strong; 2,4→weak)"
                print(f"  - {col}: {meaning}")
        
        print(f"Sample size after filtering: {len(X)}")
        
        # Feature relationship analysis
        corr_matrix, cov_matrix, high_corr_pairs = self.analyze_feature_relationships(
            X, y, period_name, save_dir)
        
        X_original, X_pca, scaler, pca = self.create_features(X)
        
        # Model definition
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5)
        }
        
        # K-fold CV setup
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {
            'period': period_name,
            'original_results': {},
            'pca_results': {},
            'regression_equations': {},
            'correlation_matrix': corr_matrix,
            'covariance_matrix': cov_matrix,
            'high_corr_pairs': high_corr_pairs,
            'data_info': {
                'n_samples': len(data),
                'n_features_original': X_original.shape[1],
                'n_features_pca': X_pca.shape[1],
                'pca_variance_explained': pca.explained_variance_ratio_.sum()
            }
        }
        
        # Original Features analysis
        print(f"\n--- Original Features Analysis ---")
        for model_name, model in models.items():
            print(f"  Analyzing {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_original, y, cv=kfold, 
                                      scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, X_original, y, cv=kfold, 
                                      scoring='r2')
            
            # Predictions (cross-validation)
            y_pred_cv = cross_val_predict(model, X_original, y, cv=kfold)
            
            # Model training (for coefficients)
            model.fit(X_original, y)
            
            # Create regression equation
            equation = self.create_regression_equation(model, feature_cols, model_name, period_name)
            
            results['original_results'][model_name] = {
                'mse_scores': -cv_scores,
                'r2_scores': r2_scores,
                'mean_mse': -cv_scores.mean(),
                'std_mse': cv_scores.std(),
                'mean_r2': r2_scores.mean(),
                'std_r2': r2_scores.std(),
                'y_true': y,
                'y_pred': y_pred_cv,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'feature_importance': np.abs(model.coef_)
            }
            
            results['regression_equations'][f'{model_name}_original'] = equation
            
            print(f"    MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"    R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
            print(equation)
        
        # PCA Components analysis
        print(f"\n--- PCA Components Analysis ---")
        for model_name, model in models.items():
            print(f"  Analyzing {model_name}...")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_pca, y, cv=kfold, 
                                      scoring='neg_mean_squared_error')
            r2_scores = cross_val_score(model, X_pca, y, cv=kfold, 
                                      scoring='r2')
            
            # Predictions (cross-validation)
            y_pred_cv = cross_val_predict(model, X_pca, y, cv=kfold)
            
            # Model training
            model.fit(X_pca, y)
            
            results['pca_results'][model_name] = {
                'mse_scores': -cv_scores,
                'r2_scores': r2_scores,
                'mean_mse': -cv_scores.mean(),
                'std_mse': cv_scores.std(),
                'mean_r2': r2_scores.mean(),
                'std_r2': r2_scores.std(),
                'y_true': y,
                'y_pred': y_pred_cv,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'feature_importance': np.abs(model.coef_)
            }
            
            print(f"    MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"    R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        
        # Save regression equations to file
        self.save_regression_equations(results['regression_equations'], period_name, save_dir)
        
        # Store results
        results['feature_names'] = feature_cols
        results['pca_components'] = X_pca.columns.tolist()
        
        self.results[period_name] = results
        return results
    
    def save_regression_equations(self, equations, period_name, save_dir):
        """Save regression equations to text file"""
        filename = f"{save_dir}/regression_equations_{period_name.replace(' ', '_')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== {period_name} Multiple Regression Analysis Results (Odd Years Only) ===\n\n")
            f.write("Note: Analysis includes only odd years (2017, 2019, 2023)\n")
            f.write("2023 Place_Attachment values were reversed during processing\n")
            f.write("Settlement_Mindset mapping: 1,3→strong(4); 2,4→weak(1-2)\n\n")
            
            for equation_name, equation in equations.items():
                f.write(f"{equation}\n")
                f.write("="*60 + "\n\n")
        
        print(f"Regression equations saved: {filename}")
    
    def create_visualizations(self, save_dir="regression_plots"):
        """Create and save visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Performance comparison
        self.plot_performance_comparison(save_dir)
        
        # 2. Feature importance
        self.plot_feature_importance(save_dir)
        
        # 3. Prediction scatter plots
        self.plot_prediction_scatter(save_dir)
        
        # 4. Before-after comparison
        self.plot_before_after_comparison(save_dir)
        
        print(f"\nAll visualizations saved to {save_dir} folder!")
    
    def plot_performance_comparison(self, save_dir):
        """Performance comparison chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        # MSE comparison - Original Features
        mse_original = []
        for period in periods:
            mse_row = [self.results[period]['original_results'][model]['mean_mse'] 
                      for model in models]
            mse_original.append(mse_row)
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, period in enumerate(periods):
            ax1.bar(x + i*width, mse_original[i], width, label=period, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE Comparison - Original Features (Odd Years)')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # R² comparison - Original Features
        r2_original = []
        for period in periods:
            r2_row = [self.results[period]['original_results'][model]['mean_r2'] 
                     for model in models]
            r2_original.append(r2_row)
        
        for i, period in enumerate(periods):
            ax2.bar(x + i*width, r2_original[i], width, label=period, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R²')
        ax2.set_title('R² Comparison - Original Features (Odd Years)')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MSE comparison - PCA
        mse_pca = []
        for period in periods:
            mse_row = [self.results[period]['pca_results'][model]['mean_mse'] 
                      for model in models]
            mse_pca.append(mse_row)
        
        for i, period in enumerate(periods):
            ax3.bar(x + i*width, mse_pca[i], width, label=period, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('MSE')
        ax3.set_title('MSE Comparison - PCA Components (Odd Years)')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # R² comparison - PCA
        r2_pca = []
        for period in periods:
            r2_row = [self.results[period]['pca_results'][model]['mean_r2'] 
                     for model in models]
            r2_pca.append(r2_row)
        
        for i, period in enumerate(periods):
            ax4.bar(x + i*width, r2_pca[i], width, label=period, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R²')
        ax4.set_title('R² Comparison - PCA Components (Odd Years)')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison_odd_years.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, save_dir):
        """Feature importance charts"""
        periods = list(self.results.keys())
        
        # Check if we have both before and after periods
        if len(periods) < 2:
            print("Need at least 2 periods for comparison")
            return
        
        # Get period names
        before_period = periods[0]  # First period
        after_period = periods[1]   # Second period
        
        # Feature mapping
        feature_mapping = {
            '만나이': 'Age',
            '지역거주기간': 'Residence_Period',
            '향후거주의향': 'Future_Residence_Intent',
            '정주의식': 'Settlement_Mindset',
            '거주지소속감': 'Place_Attachment',
            '월평균가구소득': 'Monthly_Income',
            '부채유무': 'Has_Debt',
            '삶의만족도': 'Life_Satisfaction'
        }
        
        # Get common features between periods
        before_features = self.results[before_period]['feature_names']
        after_features = self.results[after_period]['feature_names']
        common_features = list(set(before_features) & set(after_features))
        
        if not common_features:
            print("No common features found between periods.")
            return
        
        # Coefficient comparison for Ridge model
        before_coef = self.results[before_period]['original_results']['Ridge']['coefficients']
        after_coef = self.results[after_period]['original_results']['Ridge']['coefficients']
        
        # Get coefficients for common features only
        before_feature_coef = {}
        after_feature_coef = {}
        
        for i, feature in enumerate(before_features):
            if feature in common_features:
                before_feature_coef[feature] = before_coef[i]
        
        for i, feature in enumerate(after_features):
            if feature in common_features:
                after_feature_coef[feature] = after_coef[i]
        
        # Create comparison dataframe
        comparison_data = []
        for feature in common_features:
            eng_name = feature_mapping.get(feature, feature)
            comparison_data.append({
                'feature': eng_name,
                'before': before_feature_coef.get(feature, 0),
                'after': after_feature_coef.get(feature, 0),
                'change': after_feature_coef.get(feature, 0) - before_feature_coef.get(feature, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Coefficient comparison
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax1.bar(x - width/2, comparison_df['before'], width, label=before_period, alpha=0.8)
        ax1.bar(x + width/2, comparison_df['after'], width, label=after_period, alpha=0.8)
        
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Coefficient Comparison - Before vs After Redevelopment (Odd Years)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Coefficient change
        colors = ['red' if x < 0 else 'blue' for x in comparison_df['change']]
        bars = ax2.bar(x, comparison_df['change'], color=colors, alpha=0.7)
        
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Coefficient Change (After - Before)')
        ax2.set_title('Coefficient Change Due to Redevelopment (Odd Years)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on change bars
        for bar, val in zip(bars, comparison_df['change']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                    f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/coefficient_comparison_odd_years.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_scatter(self, save_dir):
        """Prediction vs Actual scatter plots"""
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        fig, axes = plt.subplots(len(periods), len(models), figsize=(15, 10))
        if len(periods) == 1:
            axes = axes.reshape(1, -1)
        
        for i, period in enumerate(periods):
            for j, model in enumerate(models):
                ax = axes[i, j]
                
                # Get data
                y_true = self.results[period]['original_results'][model]['y_true']
                y_pred = self.results[period]['original_results'][model]['y_pred']
                
                # Scatter plot
                ax.scatter(y_true, y_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                
                # Labels and title
                ax.set_xlabel('Actual Residence Satisfaction')
                ax.set_ylabel('Predicted Residence Satisfaction')
                ax.set_title(f'{period} - {model}')
                ax.grid(True, alpha=0.3)
                
                # R² score
                r2 = self.results[period]['original_results'][model]['mean_r2']
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_scatter_odd_years.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_before_after_comparison(self, save_dir):
        """Before-after comparison summary"""
        periods = list(self.results.keys())
        
        if len(periods) < 2:
            print("Need at least 2 periods for before-after comparison")
            return
        
        before_period = periods[0]
        after_period = periods[1]
        
        # Create summary comparison
        models = ['Ridge', 'Lasso', 'ElasticNet']
        metrics = ['MSE', 'R²']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original Features - MSE
        before_mse = [self.results[before_period]['original_results'][model]['mean_mse'] for model in models]
        after_mse = [self.results[after_period]['original_results'][model]['mean_mse'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, before_mse, width, label=before_period, alpha=0.8)
        ax1.bar(x + width/2, after_mse, width, label=after_period, alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE Comparison - Original Features')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Original Features - R²
        before_r2 = [self.results[before_period]['original_results'][model]['mean_r2'] for model in models]
        after_r2 = [self.results[after_period]['original_results'][model]['mean_r2'] for model in models]
        
        ax2.bar(x - width/2, before_r2, width, label=before_period, alpha=0.8)
        ax2.bar(x + width/2, after_r2, width, label=after_period, alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R²')
        ax2.set_title('R² Comparison - Original Features')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PCA Features - MSE
        before_mse_pca = [self.results[before_period]['pca_results'][model]['mean_mse'] for model in models]
        after_mse_pca = [self.results[after_period]['pca_results'][model]['mean_mse'] for model in models]
        
        ax3.bar(x - width/2, before_mse_pca, width, label=before_period, alpha=0.8)
        ax3.bar(x + width/2, after_mse_pca, width, label=after_period, alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('MSE')
        ax3.set_title('MSE Comparison - PCA Components')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # PCA Features - R²
        before_r2_pca = [self.results[before_period]['pca_results'][model]['mean_r2'] for model in models]
        after_r2_pca = [self.results[after_period]['pca_results'][model]['mean_r2'] for model in models]
        
        ax4.bar(x - width/2, before_r2_pca, width, label=before_period, alpha=0.8)
        ax4.bar(x + width/2, after_r2_pca, width, label=after_period, alpha=0.8)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R²')
        ax4.set_title('R² Comparison - PCA Components')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/before_after_comparison_odd_years.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\n=== Before-After Comparison Summary (Odd Years) ===")
        print(f"Sample sizes:")
        print(f"  {before_period}: {self.results[before_period]['data_info']['n_samples']} samples")
        print(f"  {after_period}: {self.results[after_period]['data_info']['n_samples']} samples")
        
        print(f"\nBest performing model (Original Features):")
        best_before = max(models, key=lambda m: self.results[before_period]['original_results'][m]['mean_r2'])
        best_after = max(models, key=lambda m: self.results[after_period]['original_results'][m]['mean_r2'])
        
        print(f"  {before_period}: {best_before} (R² = {self.results[before_period]['original_results'][best_before]['mean_r2']:.4f})")
        print(f"  {after_period}: {best_after} (R² = {self.results[after_period]['original_results'][best_after]['mean_r2']:.4f})")

# Usage function
def run_complete_analysis():
    """Run complete analysis for odd years only"""
    
    # Load data
    before = pd.read_csv("/content/drive/MyDrive/2025-1학기 데이터과학 1조/Data/성남시 사회조사/최종 파일/redevelopment_before_2017_2019.csv")
    after = pd.read_csv("/content/drive/MyDrive/2025-1학기 데이터과학 1조/Data/성남시 사회조사/최종 파일/redevelopment_after_2023.csv")
    
    # Filter for 65+ years old
    before = before[before['만나이'] >= 65]
    after = after[after['만나이'] >= 65]
    
    print(f"Before redevelopment (65+): {len(before)} samples")
    print(f"After redevelopment (65+): {len(after)} samples")
    
    # Check for odd years only
    if 'year' in before.columns:
        odd_before = before[before['year'].isin([2017, 2019])].copy()
        print(f"Before redevelopment (odd years only): {len(odd_before)} samples")
        before = odd_before
    
    if 'year' in after.columns:
        odd_after = after[after['year'].isin([2023])].copy()
        print(f"After redevelopment (odd years only): {len(odd_after)} samples")
        after = odd_after
    
    # Initialize analyzer
    analyzer = SeongnamOddYearAnalyzer(before, after)
    
    # Create save directory
    save_dir = "seongnam_odd_years_results"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Run regression analysis
    analyzer.run_regression_analysis("Before Redevelopment", before, save_dir)
    analyzer.run_regression_analysis("After Redevelopment", after, save_dir)
    
    # Create visualizations
    analyzer.create_visualizations(save_dir)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved in '{save_dir}' folder:")
    print("- Regression equations: regression_equations_*.txt")
    print("- Visualizations: *.png")
    print("\nSpecial Notes:")
    print("- Analysis limited to odd years only (2017, 2019, 2023)")
    print("- 2023 Place_Attachment values were reversed during processing")
    print("- Settlement_Mindset mapping: 1,3→strong(4); 2,4→weak(1-2)")
    print("- Only community-related features were analyzed (no medical features)")
    
    return analyzer

if __name__ == "__main__":
    analyzer = run_complete_analysis()