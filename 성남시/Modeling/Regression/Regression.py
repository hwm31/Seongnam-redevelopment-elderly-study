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

# Set English plotting environment
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class RegressionAnalyzer:
    def __init__(self, before_data, after_data):
        self.before_data = before_data
        self.after_data = after_data
        self.results = {}
        
    def prepare_data(self, data, target_col='거주지만족도'):
        """Data preparation without preprocessing (data already preprocessed)"""
        # Separate features and target
        feature_cols = [col for col in data.columns if col not in [target_col, 'year']]
        X = data[feature_cols]
        y = data[target_col]
        
        return X, y, feature_cols
    
    def analyze_feature_relationships(self, X, period_name, save_dir):
        """Analyze correlation and covariance between features"""
        print(f"\n=== Feature Relationship Analysis for {period_name} ===")
        
        # Calculate correlation matrix
        correlation_matrix = X.corr()
        
        # Calculate covariance matrix  
        covariance_matrix = X.cov()
        
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
        
        print(f"Number of highly correlated pairs (|r| > 0.7): {len(high_corr_pairs)}")
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .8})
        plt.title(f'Feature Correlation Matrix - {period_name}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_matrix_{period_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize covariance matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(covariance_matrix, annot=True, fmt='.2f', 
                   cmap='viridis', square=True, cbar_kws={"shrink": .8})
        plt.title(f'Feature Covariance Matrix - {period_name}')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/covariance_matrix_{period_name.replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot correlation distribution
        plt.figure(figsize=(10, 6))
        # Extract upper triangle correlations (excluding diagonal)
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        correlations = upper_triangle.stack().values
        
        plt.hist(correlations, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.7, color='red', linestyle='--', label='High Correlation Threshold (0.7)')
        plt.axvline(x=-0.7, color='red', linestyle='--')
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Feature Correlations - {period_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_distribution_{period_name.replace(" ", "_")}.png', 
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
    
    def run_regression_analysis(self, period_name, data, save_dir="regression_plots"):
        """Run regression analysis"""
        print(f"\n=== {period_name} Regression Analysis Started ===")
        
        # Data preparation
        X, y, feature_cols = self.prepare_data(data)
        
        # Feature relationship analysis
        corr_matrix, cov_matrix, high_corr_pairs = self.analyze_feature_relationships(
            X, period_name, save_dir)
        
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
            'feature_importance': {},
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
            
            # Model training (for feature importance)
            model.fit(X_original, y)
            
            results['original_results'][model_name] = {
                'mse_scores': -cv_scores,
                'r2_scores': r2_scores,
                'mean_mse': -cv_scores.mean(),
                'std_mse': cv_scores.std(),
                'mean_r2': r2_scores.mean(),
                'std_r2': r2_scores.std(),
                'y_true': y,
                'y_pred': y_pred_cv,
                'feature_importance': np.abs(model.coef_) if hasattr(model, 'coef_') else None
            }
            
            print(f"    MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"    R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        
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
                'feature_importance': np.abs(model.coef_) if hasattr(model, 'coef_') else None
            }
            
            print(f"    MSE: {-cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"    R²: {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
        
        # Feature importance (for Original Features)
        results['feature_names'] = feature_cols
        results['pca_components'] = X_pca.columns.tolist()
        
        self.results[period_name] = results
        return results
    
    def create_visualizations(self, save_dir="regression_plots"):
        """Create and save visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Performance comparison chart
        self.plot_performance_comparison(save_dir)
        
        # 2. Prediction vs Actual scatter plots
        self.plot_prediction_scatter(save_dir)
        
        # 3. Feature importance charts
        self.plot_feature_importance(save_dir)
        
        # 4. Model performance distribution
        self.plot_performance_distribution(save_dir)
        
        # 5. Before-after comparison
        self.plot_before_after_comparison(save_dir)
        
        # 6. Feature correlation comparison
        self.plot_correlation_comparison(save_dir)
        
        print(f"\nAll visualizations saved to {save_dir} folder!")
    
    def plot_performance_comparison(self, save_dir):
        """Performance comparison chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Data preparation
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        # 1. Original Features MSE comparison
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
        ax1.set_title('MSE Comparison - Original Features')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. PCA Features MSE comparison
        mse_pca = []
        for period in periods:
            mse_row = [self.results[period]['pca_results'][model]['mean_mse'] 
                      for model in models]
            mse_pca.append(mse_row)
        
        for i, period in enumerate(periods):
            ax2.bar(x + i*width, mse_pca[i], width, label=period, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('MSE')
        ax2.set_title('MSE Comparison - PCA Components')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Original Features R² comparison
        r2_original = []
        for period in periods:
            r2_row = [self.results[period]['original_results'][model]['mean_r2'] 
                     for model in models]
            r2_original.append(r2_row)
        
        for i, period in enumerate(periods):
            ax3.bar(x + i*width, r2_original[i], width, label=period, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('R²')
        ax3.set_title('R² Comparison - Original Features')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. PCA Features R² comparison
        r2_pca = []
        for period in periods:
            r2_row = [self.results[period]['pca_results'][model]['mean_r2'] 
                     for model in models]
            r2_pca.append(r2_row)
        
        for i, period in enumerate(periods):
            ax4.bar(x + i*width, r2_pca[i], width, label=period, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R²')
        ax4.set_title('R² Comparison - PCA Components')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_scatter(self, save_dir):
        """Prediction vs Actual scatter plots"""
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        # Original Features
        fig, axes = plt.subplots(len(periods), len(models), figsize=(15, 10))
        if len(periods) == 1:
            axes = axes.reshape(1, -1)
        
        for i, period in enumerate(periods):
            for j, model in enumerate(models):
                result = self.results[period]['original_results'][model]
                y_true = result['y_true']
                y_pred = result['y_pred']
                
                axes[i, j].scatter(y_true, y_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                axes[i, j].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                # R² display
                r2 = result['mean_r2']
                axes[i, j].text(0.05, 0.95, f'R² = {r2:.3f}', 
                               transform=axes[i, j].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                axes[i, j].set_xlabel('Actual')
                axes[i, j].set_ylabel('Predicted')
                axes[i, j].set_title(f'{period} - {model}')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_scatter_original.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PCA Features
        fig, axes = plt.subplots(len(periods), len(models), figsize=(15, 10))
        if len(periods) == 1:
            axes = axes.reshape(1, -1)
        
        for i, period in enumerate(periods):
            for j, model in enumerate(models):
                result = self.results[period]['pca_results'][model]
                y_true = result['y_true']
                y_pred = result['y_pred']
                
                axes[i, j].scatter(y_true, y_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                axes[i, j].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                # R² display
                r2 = result['mean_r2']
                axes[i, j].text(0.05, 0.95, f'R² = {r2:.3f}', 
                               transform=axes[i, j].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                axes[i, j].set_xlabel('Actual')
                axes[i, j].set_ylabel('Predicted')
                axes[i, j].set_title(f'{period} - {model}')
                axes[i, j].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_scatter_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, save_dir):
        """Feature importance charts"""
        periods = list(self.results.keys())
        
        for period in periods:
            # Use Ridge model's feature importance (most stable)
            importance = self.results[period]['original_results']['Ridge']['feature_importance']
            feature_names = self.results[period]['feature_names']
            
            # Top 10 features
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True).tail(10)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Feature Importance (|Coefficient|)')
            plt.title(f'Top 10 Feature Importance - {period} (Ridge)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_importance_{period.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_performance_distribution(self, save_dir):
        """Performance distribution box plots"""
        periods = list(self.results.keys())
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        # MSE distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original Features MSE
        mse_data = []
        labels = []
        for period in periods:
            for model in models:
                mse_scores = self.results[period]['original_results'][model]['mse_scores']
                mse_data.append(mse_scores)
                labels.append(f'{period}\n{model}')
        
        ax1.boxplot(mse_data, labels=labels)
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE Distribution - Original Features')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # PCA Features MSE
        mse_data_pca = []
        for period in periods:
            for model in models:
                mse_scores = self.results[period]['pca_results'][model]['mse_scores']
                mse_data_pca.append(mse_scores)
        
        ax2.boxplot(mse_data_pca, labels=labels)
        ax2.set_ylabel('MSE')
        ax2.set_title('MSE Distribution - PCA Components')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_before_after_comparison(self, save_dir):
        """Before-after comparison"""
        if len(self.results) < 2:
            print("Before-after comparison requires data from 2 or more periods.")
            return
        
        periods = list(self.results.keys())
        before_period = periods[0]
        after_period = periods[1]
        
        # Performance changes
        models = ['Ridge', 'Lasso', 'ElasticNet']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. MSE change (Original)
        mse_before = [self.results[before_period]['original_results'][model]['mean_mse'] 
                     for model in models]
        mse_after = [self.results[after_period]['original_results'][model]['mean_mse'] 
                    for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, mse_before, width, label=before_period, alpha=0.8)
        ax1.bar(x + width/2, mse_after, width, label=after_period, alpha=0.8)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('MSE')
        ax1.set_title('MSE Change - Original Features')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. R² change (Original)
        r2_before = [self.results[before_period]['original_results'][model]['mean_r2'] 
                    for model in models]
        r2_after = [self.results[after_period]['original_results'][model]['mean_r2'] 
                   for model in models]
        
        ax2.bar(x - width/2, r2_before, width, label=before_period, alpha=0.8)
        ax2.bar(x + width/2, r2_after, width, label=after_period, alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('R²')
        ax2.set_title('R² Change - Original Features')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MSE change (PCA)
        mse_before_pca = [self.results[before_period]['pca_results'][model]['mean_mse'] 
                         for model in models]
        mse_after_pca = [self.results[after_period]['pca_results'][model]['mean_mse'] 
                        for model in models]
        
        ax3.bar(x - width/2, mse_before_pca, width, label=before_period, alpha=0.8)
        ax3.bar(x + width/2, mse_after_pca, width, label=after_period, alpha=0.8)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('MSE')
        ax3.set_title('MSE Change - PCA Components')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. R² change (PCA)
        r2_before_pca = [self.results[before_period]['pca_results'][model]['mean_r2'] 
                        for model in models]
        r2_after_pca = [self.results[after_period]['pca_results'][model]['mean_r2'] 
                       for model in models]
        
        ax4.bar(x - width/2, r2_before_pca, width, label=before_period, alpha=0.8)
        ax4.bar(x + width/2, r2_after_pca, width, label=after_period, alpha=0.8)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('R²')
        ax4.set_title('R² Change - PCA Components')
        ax4.set_xticks(x)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/before_after_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_comparison(self, save_dir):
        """Feature correlation comparison between periods"""
        if len(self.results) < 2:
            return
        
        periods = list(self.results.keys())
        
        # Compare number of high correlation pairs
        high_corr_counts = []
        period_names = []
        
        for period in periods:
            high_corr_pairs = self.results[period]['high_corr_pairs']
            high_corr_counts.append(len(high_corr_pairs))
            period_names.append(period)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(period_names, high_corr_counts, alpha=0.8, color=['skyblue', 'orange'])
        plt.ylabel('Number of High Correlation Pairs (|r| > 0.7)')
        plt.title('Feature Correlation Comparison Between Periods')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, high_corr_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/correlation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_correlation_summary(self):
        """Print correlation analysis summary"""
        print("\n" + "="*60)
        print("FEATURE CORRELATION ANALYSIS SUMMARY")
        print("="*60)
        
        for period_name, results in self.results.items():
            print(f"\n--- {period_name} ---")
            print(f"Total features: {results['data_info']['n_features_original']}")
            print(f"High correlation pairs (|r| > 0.7): {len(results['high_corr_pairs'])}")
            
            if results['high_corr_pairs']:
                print("Top 5 highly correlated pairs:")
                sorted_pairs = sorted(results['high_corr_pairs'], 
                                    key=lambda x: abs(x['Correlation']), reverse=True)
                for i, pair in enumerate(sorted_pairs[:5]):
                    print(f"  {i+1}. {pair['Feature1']} - {pair['Feature2']}: {pair['Correlation']:.3f}")

# Usage function
def run_complete_analysis():
    """Run complete analysis"""
    
    # Load data
    before = pd.read_csv("성남시/Data/redevelopment_before_2017_2019.csv")
    after = pd.read_csv("성남시/Data/redevelopment_after_2023.csv")
    
    # Filter for 65+ years old
    before = before[before['만나이'] >= 65]
    after = after[after['만나이'] >= 65]
    
    print(f"Before redevelopment (65+): {len(before)} samples")
    print(f"After redevelopment (65+): {len(after)} samples")
    
    # Initialize analyzer
    analyzer = RegressionAnalyzer(before, after)
    
    # Create save directory
    save_dir = "regression_analysis_results"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Run regression analysis
    analyzer.run_regression_analysis("Before Redevelopment", before, save_dir)
    analyzer.run_regression_analysis("After Redevelopment", after, save_dir)
    
    # Create visualizations
    analyzer.create_visualizations(save_dir)
    
    # Print correlation summary
    analyzer.print_correlation_summary()
    
    return analyzer

# Execute
if __name__ == "__main__":
    analyzer = run_complete_analysis()