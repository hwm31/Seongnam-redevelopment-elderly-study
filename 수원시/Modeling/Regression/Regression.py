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

# Set basic matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class SeongnamRegressionAnalyzer:
    def __init__(self, before_data, after_data):
        self.before_data = before_data
        self.after_data = after_data
        self.results = {}
        
        # Feature information
        self.feature_info = {
            '만나이': {'type': 'continuous', 'range': 'age in years', 'direction': 'context'},
            '지역거주기간': {'type': 'continuous', 'range': 'years of residence', 'direction': 'positive'},
            '향후거주의향': {'type': 'likert', 'range': '원래 1=강하게 동의~5=강하게 반대 → 변환후 5=거주의향 강함', 'direction': 'positive'},
            '정주의식': {'type': 'special_categorical', 'range': '높을수록 정착의식 강함', 'mapping': {1: 4, 2: 2, 3: 4, 4: 1}, 'direction': 'positive'},
            '거주지소속감': {'type': 'likert', 'range': '높을수록 소속감 강함', 'direction': 'positive'},
            '주거만족도': {'type': 'likert', 'range': '원래 1=매우만족~5=매우불만족 → 변환후 5=만족도 높음', 'direction': 'positive'},
            '월평균가구소득': {'type': 'ordinal', 'range': '1=<100만원 ~ 8=700만원+', 'direction': 'positive'},
            '부채유무': {'type': 'binary', 'range': '원래 1=있음,2=없음 → 변환후 1=무부채(좋음), 0=유부채(나쁨)', 'direction': 'positive'},
            '삶의만족도': {'type': 'likert', 'range': '원래 1=매우만족~5=매우불만족 → 변환후 5=만족도 높음', 'direction': 'positive'},
            '대중교통만족도': {'type': 'continuous', 'range': '높을수록 대중교통 만족도 높음 (1~5 평균값)', 'direction': 'positive'},
            '공원이용만족도': {'type': 'continuous', 'range': '높을수록 공원 만족도 높음 (1~5)', 'direction': 'positive'}
        }
        
    def create_features(self, X):
        """Create standardized features and PCA components"""
        # Standardized features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # PCA components (retain 95% variance)
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        return X_scaled, X_pca, scaler, pca

    def calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    def create_regression_equation(self, model, feature_names, model_name="", period_name=""):
        """Create multiple regression equation with sorted feature importance"""
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
            '향후거주의향': 'Future_Residence_Intent_Pos',
            '정주의식': 'Settlement_Mindset',
            '거주지소속감': 'Place_Attachment',
            '주거만족도': 'Housing_Satisfaction',
            '월평균가구소득': 'Monthly_Income',
            '부채유무': 'Debt_Free',
            '삶의만족도': 'Life_Satisfaction_Pos',
            '대중교통만족도': 'Public_Transport_Satisfaction',
            '공원이용만족도': 'Park_Usage_Satisfaction'
        }
        
        for i, (coef, feature) in enumerate(zip(coefficients, feature_names)):
            eng_feature = feature_mapping.get(feature, feature)
            if coef >= 0:
                equation += f" + {coef:.4f} × {eng_feature}"
            else:
                equation += f" - {abs(coef):.4f} × {eng_feature}"
        
        equation += "\n"
        
        # Add coefficient interpretation - SORTED BY ABSOLUTE VALUE (IMPORTANCE)
        equation += f"\n--- Coefficient Interpretation (Sorted by Importance) ---\n"
        coef_importance = [(abs(coef), feature, coef) for coef, feature in zip(coefficients, feature_names)]
        coef_importance.sort(reverse=True)  # Sort by absolute value (importance)
        
        for i, (abs_coef, feature, coef) in enumerate(coef_importance):
            eng_feature = feature_mapping.get(feature, feature)
            direction = "increases" if coef > 0 else "decreases"
            equation += f"{i+1}. {eng_feature}: {coef:.4f} (abs={abs_coef:.4f}) → 1 unit ↑ = residence satisfaction {direction}\n"
        
        return equation
    
    def run_regression_analysis(self, period_name, X, y, save_dir="regression_plots"):
        """Run regression analysis"""
        print(f"\n=== {period_name} Regression Analysis ===")
        
        # Data preparation
        feature_cols = X.columns.tolist()
        print(f"Features: {len(feature_cols)}")
        print(f"Sample size: {len(X)}")
        
        # Create features
        X_original, X_pca, scaler, pca = self.create_features(X)
        
        # Models
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.5, l1_ratio=0.5)
        }
        
        # K-fold CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {
            'period': period_name,
            'original_results': {},
            'pca_results': {},
            'regression_equations': {},
            'data_info': {
                'n_samples': len(X),
                'n_features_original': X_original.shape[1],
                'n_features_pca': X_pca.shape[1],
                'pca_variance_explained': pca.explained_variance_ratio_.sum()
            }
        }
        
        # Original Features analysis
        print(f"\n--- Original Features (Standardized) ---")
        for model_name, model in models.items():
            print(f"  {model_name}...")
            
            # Cross-validation
            cv_scores_mse = cross_val_score(model, X_original, y, cv=kfold, scoring='neg_mean_squared_error')
            cv_scores_r2 = cross_val_score(model, X_original, y, cv=kfold, scoring='r2')
            
            # Cross-validation predictions
            y_pred_cv = cross_val_predict(model, X_original, y, cv=kfold)
            
            # Model training (for coefficients)
            model.fit(X_original, y)
            
            # Create regression equation
            equation = self.create_regression_equation(model, feature_cols, model_name, period_name)
            
            # Feature importance sorted
            feature_importance_sorted = sorted(zip(np.abs(model.coef_), feature_cols), reverse=True)
            
            results['original_results'][model_name] = {
                'cv_scores_mse': -cv_scores_mse,
                'cv_scores_r2': cv_scores_r2,
                'mean_mse': -cv_scores_mse.mean(),
                'std_mse': cv_scores_mse.std(),
                'mean_r2': cv_scores_r2.mean(),
                'std_r2': cv_scores_r2.std(),
                'y_true': y,
                'y_pred': y_pred_cv,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'feature_importance': np.abs(model.coef_),
                'feature_importance_sorted': feature_importance_sorted
            }
            
            results['regression_equations'][f'{model_name}_original'] = equation
            
            print(f"    MSE: {-cv_scores_mse.mean():.4f} (±{cv_scores_mse.std():.4f})")
            print(f"    R²: {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std():.4f})")
            
            # Print top 3 most important features
            print(f"    Top 3 Important Features:")
            for i, (importance, feature) in enumerate(feature_importance_sorted[:3], 1):
                print(f"      {i}. {feature}: {importance:.4f}")
        
        # PCA Components analysis
        print(f"\n--- PCA Components ---")
        for model_name, model in models.items():
            print(f"  {model_name}...")
            
            # Cross-validation
            cv_scores_mse = cross_val_score(model, X_pca, y, cv=kfold, scoring='neg_mean_squared_error')
            cv_scores_r2 = cross_val_score(model, X_pca, y, cv=kfold, scoring='r2')
            
            # Cross-validation predictions
            y_pred_cv = cross_val_predict(model, X_pca, y, cv=kfold)
            
            # Model training
            model.fit(X_pca, y)
            
            # Feature importance sorted for PCA
            pca_importance_sorted = sorted(zip(np.abs(model.coef_), X_pca.columns), reverse=True)
            
            results['pca_results'][model_name] = {
                'cv_scores_mse': -cv_scores_mse,
                'cv_scores_r2': cv_scores_r2,
                'mean_mse': -cv_scores_mse.mean(),
                'std_mse': cv_scores_mse.std(),
                'mean_r2': cv_scores_r2.mean(),
                'std_r2': cv_scores_r2.std(),
                'y_true': y,
                'y_pred': y_pred_cv,
                'coefficients': model.coef_,
                'intercept': model.intercept_,
                'feature_importance': np.abs(model.coef_),
                'feature_importance_sorted': pca_importance_sorted
            }
            
            print(f"    MSE: {-cv_scores_mse.mean():.4f} (±{cv_scores_mse.std():.4f})")
            print(f"    R²: {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std():.4f})")
        
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
            f.write(f"=== {period_name} Multiple Regression Analysis Results ===\n\n")
            f.write("All coefficients are sorted by importance (absolute value)\n\n")
            
            for equation_name, equation in equations.items():
                f.write(f"{equation}\n")
                f.write("="*60 + "\n\n")
        
        print(f"Regression equations saved: {filename}")
    
    def create_visualizations(self, save_dir="regression_plots"):
        """Create and save visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nCreating visualizations...")
        
        try:
            self.plot_performance_comparison(save_dir)
            print(f"  ✅ Performance comparison plotted")
        except Exception as e:
            print(f"  ❌ Error plotting performance: {str(e)}")

        try:
            self.plot_feature_importance(save_dir)
            print(f"  ✅ Feature importance plotted")
        except Exception as e:
            print(f"  ❌ Error plotting feature importance: {str(e)}")

        try:
            self.plot_prediction_scatter(save_dir)
            print(f"  ✅ Prediction scatter plotted")
        except Exception as e:
            print(f"  ❌ Error plotting prediction scatter: {str(e)}")
        
        print(f"\nAll visualizations saved to {save_dir} folder!")
    
    def plot_performance_comparison(self, save_dir):
        """Plot performance comparison"""
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        metrics_names = ['mse', 'r2']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # MSE - Original Features
        ax = axes[0]
        x = np.arange(len(models))
        width = 0.35
        
        for j, period in enumerate(periods):
            values = [self.results[period]['original_results'][model]['mean_mse'] for model in models]
            bars = ax.bar(x + j*width, values, width, label=period, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('MSE')
        ax.set_title('MSE - Original Features (Standardized)')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² - Original Features
        ax = axes[1]
        for j, period in enumerate(periods):
            values = [self.results[period]['original_results'][model]['mean_r2'] for model in models]
            bars = ax.bar(x + j*width, values, width, label=period, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('R²')
        ax.set_title('R² - Original Features (Standardized)')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MSE - PCA Components
        ax = axes[2]
        for j, period in enumerate(periods):
            values = [self.results[period]['pca_results'][model]['mean_mse'] for model in models]
            bars = ax.bar(x + j*width, values, width, label=period, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('MSE')
        ax.set_title('MSE - PCA Components')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # R² - PCA Components
        ax = axes[3]
        for j, period in enumerate(periods):
            values = [self.results[period]['pca_results'][model]['mean_r2'] for model in models]
            bars = ax.bar(x + j*width, values, width, label=period, alpha=0.8)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('R²')
        ax.set_title('R² - PCA Components')
        ax.set_xticks(x + width/2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_feature_importance(self, save_dir):
        """Plot feature importance for original features"""
        periods = list(self.results.keys())
        
        if len(periods) < 2:
            print("Need at least 2 periods for comparison")
            return
        
        before_period = periods[0]
        after_period = periods[1]
        
        # Feature mapping
        feature_mapping = {
            '만나이': 'Age',
            '지역거주기간': 'Residence_Period',
            '향후거주의향': 'Future_Residence_Intent_Pos',
            '정주의식': 'Settlement_Mindset',
            '거주지소속감': 'Place_Attachment',
            '주거만족도': 'Housing_Satisfaction',
            '월평균가구소득': 'Monthly_Income',
            '부채유무': 'Debt_Free',
            '삶의만족도': 'Life_Satisfaction_Pos',
            '대중교통만족도': 'Public_Transport_Satisfaction',
            '공원이용만족도': 'Park_Usage_Satisfaction'
        }
        
        # Get sorted feature importance for Ridge model
        before_sorted = self.results[before_period]['original_results']['Ridge']['feature_importance_sorted']
        after_sorted = self.results[after_period]['original_results']['Ridge']['feature_importance_sorted']
        
        # Create comparison dataframe - use the order from the 'after' period
        after_feature_order = [feat for _, feat in after_sorted]
        
        comparison_data = []
        before_dict = dict(before_sorted)
        after_dict = dict(after_sorted)
        
        for feature in after_feature_order:
            eng_name = feature_mapping.get(feature, feature)
            comparison_data.append({
                'feature': eng_name,
                'before': before_dict.get(feature, 0),
                'after': after_dict.get(feature, 0),
                'change': after_dict.get(feature, 0) - before_dict.get(feature, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Feature importance comparison (sorted by after period importance)
        x = np.arange(len(comparison_df))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, comparison_df['before'], width, 
                       label=before_period, alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, comparison_df['after'], width, 
                       label=after_period, alpha=0.8, color='darkorange')
        
        ax1.set_xlabel('Features (Sorted by After-Period Importance)')
        ax1.set_ylabel('Feature Importance (|Coefficient|)')
        ax1.set_title('Ridge Regression Feature Importance Comparison\n(Sorted by Importance in After Period)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (before_val, after_val) in enumerate(zip(comparison_df['before'], comparison_df['after'])):
            if before_val > 0:
                ax1.text(i - width/2, before_val + 0.005, f'{before_val:.3f}',
                        ha='center', va='bottom', fontsize=8)
            if after_val > 0:
                ax1.text(i + width/2, after_val + 0.005, f'{after_val:.3f}',
                        ha='center', va='bottom', fontsize=8)
        
        # Importance change
        colors = ['red' if x < 0 else 'blue' for x in comparison_df['change']]
        bars = ax2.bar(x, comparison_df['change'], color=colors, alpha=0.7)
        
        ax2.set_xlabel('Features (Sorted by After-Period Importance)')
        ax2.set_ylabel('Importance Change (After - Before)')
        ax2.set_title('Feature Importance Change\n(Sorted by Importance in After Period)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(comparison_df['feature'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on change bars
        for bar, val in zip(bars, comparison_df['change']):
            height = bar.get_height()
            if abs(height) > 0.001:
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.002 if height >= 0 else -0.002),
                        f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_prediction_scatter(self, save_dir):
        """Prediction vs Actual scatter plots"""
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        fig, axes = plt.subplots(2, len(models), figsize=(15, 10))
        if len(periods) == 1:
            axes = axes.reshape(1, -1)
        
        # Original Features
        for j, model in enumerate(models):
            for i, period in enumerate(periods):
                if len(periods) == 1:
                    ax = axes[j]
                else:
                    ax = axes[0, j] if i == 0 else axes[1, j] if len(periods) > 1 else axes[j]
                
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
                ax.set_title(f'{period} - {model} (Original Features)')
                ax.grid(True, alpha=0.3)
                
                # R² score
                r2 = self.results[period]['original_results'][model]['mean_r2']
                ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                if len(periods) == 1:
                    break
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_results(analyzer):
    """Print regression results"""
    print(f"\n{'='*80}")
    print(f"REGRESSION ANALYSIS RESULTS")
    print(f"{'='*80}")

    periods = list(analyzer.results.keys())
    models = ['Ridge', 'Lasso', 'ElasticNet']

    for period in periods:
        result = analyzer.results[period]
        print(f"\n📊 {period}")
        print("="*60)

        data_info = result['data_info']
        print(f"📋 Data: {data_info['n_samples']} samples, "
              f"{data_info['n_features_original']} features, "
              f"{data_info['n_features_pca']} PCA components")

        print(f"\n🔍 Original Features (Standardized):")
        print(f"{'Model':<15} {'MSE':<10} {'R²':<10} {'Top Feature':<20} {'Importance':<10}")
        print("-"*75)

        for model in models:
            if model in result['original_results']:
                res = result['original_results'][model]
                top_feature = res['feature_importance_sorted'][0]
                print(f"{model:<15} {res['mean_mse']:<10.3f} "
                      f"{res['mean_r2']:<10.3f} "
                      f"{top_feature[1]:<20} {top_feature[0]:<10.3f}")
            else:
                print(f"{model:<15} {'N/A':<10} {'N/A':<10} {'N/A':<20} {'N/A':<10}")

        print(f"\n🔍 PCA Components:")
        print(f"{'Model':<15} {'MSE':<10} {'R²':<10}")
        print("-"*40)

        for model in models:
            if model in result['pca_results']:
                res = result['pca_results'][model]
                print(f"{model:<15} {res['mean_mse']:<10.3f} "
                      f"{res['mean_r2']:<10.3f}")
            else:
                print(f"{model:<15} {'N/A':<10} {'N/A':<10}")

        # Feature importance for Ridge model
        if 'Ridge' in result['original_results']:
            print(f"\n🎯 Ridge Regression Feature Importance (Top 5):")
            ridge_sorted = result['original_results']['Ridge']['feature_importance_sorted'][:5]
            for i, (importance, feature) in enumerate(ridge_sorted, 1):
                print(f"  {i}. {feature}: {importance:.4f}")

    print(f"\n{'='*80}")
    print(f"✅ Regression Analysis Complete")
    print(f"📈 Features sorted by importance (absolute coefficient values)")
    print(f"{'='*80}")

def run_complete_ensemble_analysis():
    """Run complete regression analysis"""
    
    print("🚀 Starting Regression Analysis: Original Features vs PCA Components")
    
    # Load data - adjust file paths as needed
    before = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/redevelopment_before_2017_2019.csv")
    after = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/redevelopment_after_2023.csv")
    
    print(f"Before redevelopment: {len(before)} samples")
    print(f"After redevelopment: {len(after)} samples")
    
    # Separate features and target
    target_col = '지역생활만족도'
    before_X = before.drop(target_col, axis=1)
    before_y = before[target_col]
    after_X = after.drop(target_col, axis=1)
    after_y = after[target_col]
    
    # Initialize analyzer
    analyzer = SeongnamRegressionAnalyzer(before, after)
    
    # Create save directory
    save_dir = "regression_results"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Run regression analysis
    analyzer.run_regression_analysis("Before Redevelopment", before_X, before_y, save_dir)
    analyzer.run_regression_analysis("After Redevelopment", after_X, after_y, save_dir)
    
    # Create visualizations
    analyzer.create_visualizations(save_dir)
    
    # Print results
    print_results(analyzer)
    
    print(f"\n=== Regression Analysis Complete ===")
    print(f"Results saved in '{save_dir}' folder:")
    print("- regression_equations_*.txt: Regression equations with sorted coefficients")
    print("- performance_comparison.png: MSE and R² comparison")
    print("- feature_importance_comparison.png: Feature importance sorted comparison")
    print("- prediction_scatter.png: Actual vs Predicted scatter plots")
    print("\nModels analyzed:")
    print("1. Ridge, Lasso, ElasticNet with Original Features (standardized) and K-fold CV")
    print("2. Ridge, Lasso, ElasticNet with PCA Components and K-fold CV")
    print("\nFeatures:")
    print("- All feature importance sorted by absolute coefficient values")
    print("- Top 3 most important features displayed for each model")
    
    return analyzer

if __name__ == "__main__":
    analyzer = run_complete_ensemble_analysis()