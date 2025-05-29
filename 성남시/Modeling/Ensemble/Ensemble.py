import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

class SeongnamEnsembleAnalyzer:
    def __init__(self, before_data, after_data):
        self.before_data = before_data
        self.after_data = after_data
        self.results = {}
        
        # Feature information for odd years only
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
        
    def create_elderly_friendliness_grades(self, data):
        """Create elderly friendliness grades based on residence satisfaction"""
        # Process residence satisfaction (1=very satisfied ~ 5=very dissatisfied)
        # Reverse it so higher score = higher satisfaction
        satisfaction = data['거주지만족도'].copy()
        max_val = satisfaction.max()
        satisfaction_reversed = (max_val + 1) - satisfaction
        
        # Create grades based on satisfaction levels
        # We'll use quantile-based approach for balanced classes
        grades = []
        
        # Calculate quantiles for balanced classification
        q33 = satisfaction_reversed.quantile(0.33)
        q67 = satisfaction_reversed.quantile(0.67)
        
        for score in satisfaction_reversed:
            if score <= q33:
                grades.append('Low')  # Low friendliness
            elif score <= q67:
                grades.append('Medium')  # Medium friendliness  
            else:
                grades.append('High')  # High friendliness
                
        return pd.Series(grades), satisfaction_reversed
    
    def prepare_ensemble_data(self, data):
        """Data preparation for ensemble models"""
        # Filter only odd years data
        if 'year' in data.columns:
            odd_years = [2017, 2019, 2021, 2023]
            data = data[data['year'].isin(odd_years)].copy()
        
        # Create target variable (elderly friendliness grades)
        y_grades, satisfaction_scores = self.create_elderly_friendliness_grades(data)
        
        # Separate features (exclude target variable)
        feature_cols = [col for col in data.columns if col not in ['거주지만족도', 'year'] and col in self.feature_info]
        X = data[feature_cols].copy()
        
        # Process features
        X_processed = self.process_features(X, data)
        
        return X_processed, y_grades, feature_cols, satisfaction_scores
    
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
                        if 2023 in original_data['year'].values:
                            processed_col = X[col].copy()
                            if len(original_data) == len(X):
                                year_2023_mask = (original_data['year'] == 2023)
                                processed_col[year_2023_mask] = 5 - X[col][year_2023_mask]
                            X_processed[col] = processed_col
                        else:
                            X_processed[col] = X[col]
                    
                    # Handle other likert scales
                    elif feature_info.get('reverse', False):
                        max_val = X[col].max()
                        X_processed[col] = (max_val + 1) - X[col]
                    else:
                        X_processed[col] = X[col]
                
                elif feature_type == 'binary':
                    X_processed[col] = (X[col] == 1).astype(int)
                
                elif feature_type == 'special_categorical':
                    if col == '정주의식':
                        mapping = feature_info.get('mapping', {})
                        X_processed[col] = X[col].map(mapping).fillna(X[col])
                    else:
                        X_processed[col] = pd.to_numeric(X[col], errors='coerce')
                
                elif feature_type == 'ordinal':
                    X_processed[col] = pd.to_numeric(X[col], errors='coerce')
                
                elif feature_type == 'continuous':
                    X_processed[col] = pd.to_numeric(X[col], errors='coerce')
        
        # Fill missing values with median
        X_processed = X_processed.fillna(X_processed.median())
        
        return X_processed
    
    def create_ensemble_features(self, X):
        """Create Original Features (standardized) and PCA Components"""
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
    
    def run_ensemble_analysis(self, period_name, data, save_dir="ensemble_plots"):
        """Run ensemble model analysis"""
        print(f"\n=== {period_name} Ensemble Analysis (Odd Years Only) ===")
        
        # Data preparation
        X, y, feature_cols, satisfaction_scores = self.prepare_ensemble_data(data)
        
        print(f"Features prepared: {len(feature_cols)}")
        print(f"Sample size after filtering: {len(X)}")
        
        # Create features
        X_original, X_pca, scaler, pca = self.create_ensemble_features(X)
        
        # Model definition
        base_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        }
        
        # Voting ensemble components
        voting_components = {
            'lr': LogisticRegression(random_state=42, max_iter=1000),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'dt': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        
        # Create voting ensembles
        voting_models = {
            'VotingEnsemble': VotingClassifier(
                estimators=list(voting_components.items()),
                voting='soft'
            )
        }
        
        # Combine all models
        all_models = {**base_models, **voting_models}
        
        # K-fold CV setup
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {
            'period': period_name,
            'original_results': {},
            'pca_results': {},
            'data_info': {
                'n_samples': len(data),
                'n_features_original': X_original.shape[1],
                'n_features_pca': X_pca.shape[1],
                'pca_variance_explained': pca.explained_variance_ratio_.sum()
            }
        }
        
        # Original Features analysis
        print(f"\n--- Original Features Ensemble Analysis ---")
        for model_name, model in all_models.items():
            print(f"  Analyzing {model_name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_original, y, cv=kfold, scoring='accuracy')
                
                # Predictions (cross-validation)
                y_pred_cv = cross_val_predict(model, X_original, y, cv=kfold)
                
                # Model training (for feature importance if available)
                model.fit(X_original, y)
                
                # Feature importance (for tree-based models)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
                    # For voting ensemble, get average feature importance
                    importances = []
                    for est_name, estimator in model.named_estimators_.items():
                        if hasattr(estimator, 'feature_importances_'):
                            importances.append(estimator.feature_importances_)
                    if importances:
                        feature_importance = np.mean(importances, axis=0)
                
                results['original_results'][model_name] = {
                    'cv_scores': cv_scores,
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'y_true': y,
                    'y_pred': y_pred_cv,
                    'feature_importance': feature_importance,
                    'classification_report': classification_report(y, y_pred_cv, output_dict=True),
                    'confusion_matrix': confusion_matrix(y, y_pred_cv)
                }
                
                print(f"    Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                continue
        
        # PCA Components analysis
        print(f"\n--- PCA Components Ensemble Analysis ---")
        for model_name, model in all_models.items():
            print(f"  Analyzing {model_name}...")
            
            try:
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_pca, y, cv=kfold, scoring='accuracy')
                
                # Predictions (cross-validation)
                y_pred_cv = cross_val_predict(model, X_pca, y, cv=kfold)
                
                # Model training
                model.fit(X_pca, y)
                
                # Feature importance (for tree-based models)
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
                    # For voting ensemble, get average feature importance
                    importances = []
                    for est_name, estimator in model.named_estimators_.items():
                        if hasattr(estimator, 'feature_importances_'):
                            importances.append(estimator.feature_importances_)
                    if importances:
                        feature_importance = np.mean(importances, axis=0)
                
                results['pca_results'][model_name] = {
                    'cv_scores': cv_scores,
                    'mean_accuracy': cv_scores.mean(),
                    'std_accuracy': cv_scores.std(),
                    'y_true': y,
                    'y_pred': y_pred_cv,
                    'feature_importance': feature_importance,
                    'classification_report': classification_report(y, y_pred_cv, output_dict=True),
                    'confusion_matrix': confusion_matrix(y, y_pred_cv)
                }
                
                print(f"    Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                continue
        
        # Store results
        results['feature_names'] = feature_cols
        results['pca_components'] = X_pca.columns.tolist()
        results['satisfaction_scores'] = satisfaction_scores
        
        self.results[period_name] = results
        return results
    
    def create_ensemble_visualizations(self, save_dir="ensemble_plots"):
        """Create and save ensemble visualizations"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Ensemble performance comparison
        self.plot_ensemble_performance(save_dir)
        
        # 2. Feature importance comparison
        self.plot_ensemble_feature_importance(save_dir)
        
        # 3. Model comparison with previous results
        self.plot_comprehensive_model_comparison(save_dir)
        
        # 4. Ensemble confusion matrices
        self.plot_ensemble_confusion_matrices(save_dir)
        
        print(f"\nAll ensemble visualizations saved to {save_dir} folder!")
    
    def plot_ensemble_performance(self, save_dir):
        """Plot ensemble model performance comparison"""
        periods = list(self.results.keys())
        models = ['RandomForest', 'GradientBoosting', 'VotingEnsemble']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison - Original Features
        acc_original = []
        for period in periods:
            acc_row = []
            for model in models:
                if model in self.results[period]['original_results']:
                    acc_row.append(self.results[period]['original_results'][model]['mean_accuracy'])
                else:
                    acc_row.append(0)
            acc_original.append(acc_row)
        
        x = np.arange(len(models))
        width = 0.35
        
        for i, period in enumerate(periods):
            ax1.bar(x + i*width, acc_original[i], width, label=period, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Ensemble Accuracy - Original Features')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Standard deviation - Original Features
        std_original = []
        for period in periods:
            std_row = []
            for model in models:
                if model in self.results[period]['original_results']:
                    std_row.append(self.results[period]['original_results'][model]['std_accuracy'])
                else:
                    std_row.append(0)
            std_original.append(std_row)
        
        for i, period in enumerate(periods):
            ax2.bar(x + i*width, std_original[i], width, label=period, alpha=0.8)
        
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Ensemble Accuracy Std Dev - Original Features')
        ax2.set_xticks(x + width/2)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Accuracy comparison - PCA Components
        acc_pca = []
        for period in periods:
            acc_row = []
            for model in models:
                if model in self.results[period]['pca_results']:
                    acc_row.append(self.results[period]['pca_results'][model]['mean_accuracy'])
                else:
                    acc_row.append(0)
            acc_pca.append(acc_row)
        
        for i, period in enumerate(periods):
            ax3.bar(x + i*width, acc_pca[i], width, label=period, alpha=0.8)
        
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Ensemble Accuracy - PCA Components')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Standard deviation - PCA Components
        std_pca = []
        for period in periods:
            std_row = []
            for model in models:
                if model in self.results[period]['pca_results']:
                    std_row.append(self.results[period]['pca_results'][model]['std_accuracy'])
                else:
                    std_row.append(0)
            std_pca.append(std_row)
        
        for i, period in enumerate(periods):
            ax4.bar(x + i*width, std_pca[i], width, label=period, alpha=0.8)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Ensemble Accuracy Std Dev - PCA Components')
        ax4.set_xticks(x + width/2)
        ax4.set_xticklabels(models)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ensemble_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ensemble_feature_importance(self, save_dir):
        """Plot feature importance for ensemble models"""
        periods = list(self.results.keys())
        
        if len(periods) < 2:
            print("Need at least 2 periods for feature importance comparison")
            return
        
        before_period = periods[0]
        after_period = periods[1]
        
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
        
        # Get feature importance for Random Forest (most reliable)
        try:
            before_importance = self.results[before_period]['original_results']['RandomForest']['feature_importance']
            after_importance = self.results[after_period]['original_results']['RandomForest']['feature_importance']
            
            if before_importance is None or after_importance is None:
                print("Feature importance not available for Random Forest")
                return
            
            # Get feature names
            feature_names = self.results[before_period]['feature_names']
            
            # Create comparison dataframe
            importance_data = []
            for i, feature in enumerate(feature_names):
                eng_name = feature_mapping.get(feature, feature)
                importance_data.append({
                    'feature': eng_name,
                    'before': before_importance[i] if i < len(before_importance) else 0,
                    'after': after_importance[i] if i < len(after_importance) else 0
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_df['change'] = importance_df['after'] - importance_df['before']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Feature importance comparison
            x = np.arange(len(importance_df))
            width = 0.35
            
            ax1.bar(x - width/2, importance_df['before'], width, label=before_period, alpha=0.8)
            ax1.bar(x + width/2, importance_df['after'], width, label=after_period, alpha=0.8)
            
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Feature Importance')
            ax1.set_title('Random Forest Feature Importance Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(importance_df['feature'], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Importance change
            colors = ['red' if x < 0 else 'blue' for x in importance_df['change']]
            bars = ax2.bar(x, importance_df['change'], color=colors, alpha=0.7)
            
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Importance Change (After - Before)')
            ax2.set_title('Feature Importance Change - Random Forest')
            ax2.set_xticks(x)
            ax2.set_xticklabels(importance_df['feature'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars, importance_df['change']):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.001 if height >= 0 else -0.001),
                        f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/ensemble_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error creating feature importance plot: {str(e)}")
    
    def plot_comprehensive_model_comparison(self, save_dir):
        """Plot comprehensive comparison including previous models"""
        # This would be a comprehensive view comparing all models
        # For now, we'll create a summary table
        
        print("\n=== Comprehensive Model Performance Summary ===")
        
        for period_name, results in self.results.items():
            print(f"\n{period_name}:")
            print("Original Features:")
            for model_name, model_results in results['original_results'].items():
                acc = model_results['mean_accuracy']
                std = model_results['std_accuracy']
                print(f"  {model_name}: {acc:.4f} (±{std:.4f})")
            
            print("PCA Components:")
            for model_name, model_results in results['pca_results'].items():
                acc = model_results['mean_accuracy']
                std = model_results['std_accuracy']
                print(f"  {model_name}: {acc:.4f} (±{std:.4f})")
    
    def plot_ensemble_confusion_matrices(self, save_dir):
        """Plot confusion matrices for ensemble models"""
        periods = list(self.results.keys())
        models = ['RandomForest', 'GradientBoosting', 'VotingEnsemble']
        
        fig, axes = plt.subplots(len(periods), len(models), figsize=(15, 10))
        if len(periods) == 1:
            axes = axes.reshape(1, -1)
        
        for i, period in enumerate(periods):
            for j, model in enumerate(models):
                if model in self.results[period]['original_results']:
                    ax = axes[i, j]
                    cm = self.results[period]['original_results'][model]['confusion_matrix']
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['High', 'Low', 'Medium'],
                               yticklabels=['High', 'Low', 'Medium'])
                    ax.set_title(f'{period} - {model}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ensemble_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()

# Usage function
def run_complete_ensemble_analysis():
    """Run complete ensemble analysis"""
    
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
    
    # Initialize ensemble analyzer
    analyzer = SeongnamEnsembleAnalyzer(before, after)
    
    # Create save directory
    save_dir = "seongnam_ensemble_results"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Run ensemble analysis
    analyzer.run_ensemble_analysis("Before Redevelopment", before, save_dir)
    analyzer.run_ensemble_analysis("After Redevelopment", after, save_dir)
    
    # Create visualizations
    analyzer.create_ensemble_visualizations(save_dir)
    
    print(f"\n=== Ensemble Analysis Complete ===")
    print(f"Results saved in '{save_dir}' folder:")
    print("- Ensemble performance comparison")
    print("- Feature importance analysis")
    print("- Comprehensive model comparison")
    print("- Confusion matrices")
    print("\nEnsemble Models Analyzed:")
    print("- Random Forest (Original & PCA)")
    print("- Gradient Boosting (Original & PCA)")
    print("- Voting Ensemble: LogisticRegression + KNN + DecisionTree (Original & PCA)")
    
    return analyzer

if __name__ == "__main__":
    analyzer = run_complete_ensemble_analysis()