import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set basic matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class ElderlyFriendlinessClassifier:
    def __init__(self, before_data, after_data):
        self.before_data = before_data
        self.after_data = after_data
        self.results = {}
        
        # TODO 
        # Feature information with English mapping
        self.feature_info = {
            '만나이': {'type': 'continuous', 'range': 'age in years', 'direction': 'context', 'english': 'Age'},
            '지역거주기간': {'type': 'continuous', 'range': 'years of residence', 'direction': 'positive', 'english': 'Residence_Period'},
            '향후거주의향': {'type': 'likert', 'range': '원래 1=강하게 동의~5=강하게 반대 → 변환후 5=거주의향 강함', 'direction': 'positive', 'english': 'Future_Residence_Intent'},
            '정주의식': {'type': 'special_categorical', 'range': '높을수록 정착의식 강함', 'mapping': {1: 4, 2: 2, 3: 4, 4: 1}, 'direction': 'positive', 'english': 'Settlement_Mindset'},
            '거주지소속감': {'type': 'likert', 'range': '높을수록 소속감 강함', 'direction': 'positive', 'english': 'Place_Attachment'},
            '주거만족도': {'type': 'likert', 'range': '원래 1=매우만족~5=매우불만족 → 변환후 5=만족도 높음', 'direction': 'positive', 'english': 'Housing_Satisfaction'},
            '월평균가구소득': {'type': 'ordinal', 'range': '1=<100만원 ~ 8=700만원+', 'direction': 'positive', 'english': 'Monthly_Income'},
            '부채유무': {'type': 'binary', 'range': '원래 1=있음,2=없음 → 변환후 1=무부채(좋음), 0=유부채(나쁨)', 'direction': 'positive', 'english': 'Debt_Free'},
            '삶의만족도': {'type': 'likert', 'range': '원래 1=매우만족~5=매우불만족 → 변환후 5=만족도 높음', 'direction': 'positive', 'english': 'Life_Satisfaction'},
            '대중교통만족도': {'type': 'continuous', 'range': '높을수록 대중교통 만족도 높음 (1~5 평균값)', 'direction': 'positive', 'english': 'Public_Transport_Satisfaction'},
            "공원이용만족도": {'type': 'continuous', 'range': '높을수록 공원 이용 만족도 높음 (1~5)', 'direction': 'positive', 'english': 'Park_Usage_Satisfaction'}
        }
        
    def create_elderly_friendliness_grades(self, data):
        """Create elderly friendliness grades based on regional life satisfaction"""
        # First reverse transform: 1=very satisfied -> 5=very satisfied (higher = better)
        satisfaction_raw = data['지역생활만족도'].copy()
        max_val = satisfaction_raw.max()
        satisfaction = (max_val + 1) - satisfaction_raw  # Reverse transform
    
        # Create grades based on quantiles of TRANSFORMED values
        q33 = satisfaction.quantile(0.33)
        q67 = satisfaction.quantile(0.67)
    
        grades = []
        for score in satisfaction:
            if score <= q33:
                grades.append('Low')    # Low transformed score = Low satisfaction
            elif score <= q67:
                grades.append('Medium')
            else:
                grades.append('High')   # High transformed score = High satisfaction
            
        return pd.Series(grades, index=data.index)

    def balance_classes(self, X, y, method='none', verbose=True):
        """Balance classes using SMOTE or keep original distribution"""
        if verbose:
            print(f"\nOriginal class distribution:")
            original_counts = pd.Series(y).value_counts()
            for cls, count in original_counts.items():
                print(f"  {cls}: {count}")

        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            method_name = "SMOTE Oversampling"
        else:
            X_balanced, y_balanced = X, y
            method_name = "Original Distribution"

        if verbose:
            print(f"\nAfter {method_name}:")
            balanced_counts = pd.Series(y_balanced).value_counts()
            for cls, count in balanced_counts.items():
                print(f"  {cls}: {count}")

        return X_balanced, y_balanced, method_name
    
    def prepare_classification_data(self, data):
        """Data preparation for classification with proper feature processing"""
        # Create target variable
        y = self.create_elderly_friendliness_grades(data)
        
        # Select features that exist in both data and feature_info
        feature_cols = [col for col in data.columns if col not in ['지역생활만족도', 'year'] and col in self.feature_info]
        X = data[feature_cols].copy()
        
        # Process features with proper transformations
        X_processed = self.process_features(X, data)
        
        # Convert to English column names
        X_processed_english = X_processed.copy()
        english_feature_cols = []
        for col in feature_cols:
            if col in self.feature_info:
                english_name = self.feature_info[col]['english']
                X_processed_english = X_processed_english.rename(columns={col: english_name})
                english_feature_cols.append(english_name)
        
        return X_processed_english, y, english_feature_cols
    
    def process_features(self, X, original_data):
        """Process features with year-specific handling - ALL FEATURES MADE POSITIVE DIRECTION"""
        X_processed = X.copy()

        for col in X.columns:
            if col in self.feature_info:
                feature_info = self.feature_info[col]
                feature_type = feature_info['type']

                if feature_type == 'likert':
                    # 모든 likert scale 변수를 높을수록 긍정적으로 변환
                    if col == '거주지소속감':
                        # 거주지소속감: 원래 1=없음~4=매우강함 (이미 높을수록 좋음)
                        if 'year' in original_data.columns and 2023 in original_data['year'].values:
                            # 2023년 데이터는 reverse되어 있으므로 다시 reverse
                            processed_col = X[col].copy()
                            if len(original_data) == len(X):
                                year_2023_mask = (original_data['year'] == 2023)
                                processed_col[year_2023_mask] = 5 - X[col][year_2023_mask]
                            X_processed[col] = processed_col
                        else:
                            X_processed[col] = X[col]  # 원래 방향 유지
                    elif col in ['향후거주의향', '주거만족도', '삶의만족도', '공원이용만족도']:
                        # 만족도 관련 변수들을 높을수록 좋게 변환
                        max_val = X[col].max()
                        X_processed[col] = (max_val + 1) - X[col]
                    else:
                        X_processed[col] = X[col]

                elif feature_type == 'binary':
                    if col == '부채유무':
                        # 부채유무: 1=있음, 2=없음 → 1=없음(좋음), 0=있음(나쁨)으로 변환
                        X_processed[col] = (X[col] == 2).astype(int)  # 2(없음)→1, 1(있음)→0
                    else:
                        X_processed[col] = (X[col] == 1).astype(int)

                elif feature_type == 'special_categorical':
                    if col == '정주의식':
                        # 정주의식 매핑은 이미 높을수록 좋게 설정되어 있음
                        mapping = feature_info.get('mapping', {})
                        X_processed[col] = X[col].map(mapping).fillna(X[col])
                    else:
                        X_processed[col] = pd.to_numeric(X[col], errors='coerce')

                elif feature_type in ['ordinal', 'continuous']:
                    # 연속형/순서형 변수는 원래 방향 유지 (이미 높을수록 좋음)
                    X_processed[col] = pd.to_numeric(X[col], errors='coerce')

        X_processed = X_processed.fillna(X_processed.median())

        print(f"\n✅ All features processed to POSITIVE direction (higher = better):")
        print(f"   - Future_Residence_Intent: Higher = Stronger residence intention")
        print(f"   - Housing_Satisfaction: Higher = Higher housing satisfaction")
        print(f"   - Life_Satisfaction: Higher = Higher life satisfaction")
        print(f"   - Place_Attachment: Higher = Stronger place attachment")
        print(f"   - Settlement_Mindset: Higher = Stronger settlement mindset")
        print(f"   - Debt_Free: 1=No debt(good), 0=Has debt(bad)")
        print(f"   - Age, Residence_Period, Monthly_Income: Original direction maintained")

        return X_processed
    
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

    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate classification metrics including ROC/AUC"""
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=['Low', 'Medium', 'High'], zero_division=0)
        
        cm = confusion_matrix(y_true, y_pred, labels=['Low', 'Medium', 'High'])
        
        # Calculate ROC/AUC if probabilities are available
        roc_auc_results = None
        if y_pred_proba is not None:
            roc_auc_results = self.calculate_roc_auc(y_true, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class,
            'confusion_matrix': cm,
            'roc_auc_results': roc_auc_results
        }

    def calculate_roc_auc(self, y_true, y_pred_proba):
        """Calculate ROC curves and AUC scores for multiclass classification"""
        from sklearn.preprocessing import LabelBinarizer
        from sklearn.metrics import roc_curve, auc
        
        classes = ['Low', 'Medium', 'High']
        n_classes = len(classes)

        try:
            label_binarizer = LabelBinarizer()
            y_true_binary = label_binarizer.fit_transform(y_true)

            if y_true_binary.shape[1] == 1:
                y_true_binary = np.column_stack([1 - y_true_binary, y_true_binary])

            if y_pred_proba.shape[1] != y_true_binary.shape[1]:
                # 클래스 수 맞춤
                n_pred_classes = y_pred_proba.shape[1]
                if n_pred_classes == 3 and y_true_binary.shape[1] == 2:
                    # 3클래스 예측을 위한 조정
                    y_true_binary_adjusted = np.zeros((len(y_true), 3))
                    for i, label in enumerate(y_true):
                        if label == 'Low':
                            y_true_binary_adjusted[i, 0] = 1
                        elif label == 'Medium':
                            y_true_binary_adjusted[i, 1] = 1
                        else:  # High
                            y_true_binary_adjusted[i, 2] = 1
                    y_true_binary = y_true_binary_adjusted

            fpr = {}
            tpr = {}
            roc_auc = {}

            for i in range(min(n_classes, y_true_binary.shape[1], y_pred_proba.shape[1])):
                fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Macro average
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(min(n_classes, len(fpr)))]))
            mean_tpr = np.zeros_like(all_fpr)

            for i in range(min(n_classes, len(fpr))):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= min(n_classes, len(fpr))

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            return {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'classes': classes
            }
        except Exception as e:
            print(f"Error calculating ROC/AUC: {str(e)}")
            return None
    
    def run_classification_analysis(self, period_name, data, balance_method='none'):
        """Run classification analysis"""
        print(f"\n=== {period_name} Classification Analysis ({balance_method.upper()}) ===")
        
        # Data preparation
        X, y, feature_cols = self.prepare_classification_data(data)
        print(f"Features: {len(feature_cols)}")
        print(f"Sample size: {len(X)}")
        
        # Create features
        X_original, X_pca, scaler, pca = self.create_features(X)
        
        # Apply balancing
        y_original = y.copy()
        X_original, y, balance_method_name = self.balance_classes(X_original, y, balance_method, verbose=True)
        X_pca, y_pca, _ = self.balance_classes(X_pca, y_original, balance_method, verbose=False)
        
        # Models
        models = {
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # K-fold CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {
            'period': period_name,
            'balance_method': balance_method,
            'balance_method_name': balance_method_name,
            'original_results': {},
            'pca_results': {},
            'data_info': {
                'n_samples': len(data),
                'n_samples_processed': len(X_original),
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
            cv_scores = cross_val_score(model, X_original, y, cv=kfold, scoring='accuracy')
            y_pred_cv = cross_val_predict(model, X_original, y, cv=kfold)
            
            # Get probability predictions for ROC/AUC
            y_pred_proba_cv = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba_cv = cross_val_predict(model, X_original, y, cv=kfold, method='predict_proba')
                except Exception as e:
                    print(f"    Warning: Could not get probability predictions for {model_name}: {str(e)}")
            
            # Calculate metrics
            metrics = self.calculate_metrics(y, y_pred_cv, y_pred_proba_cv)
            
            # Feature importance for Decision Tree
            model.fit(X_original, y)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            
            results['original_results'][model_name] = {
                'cv_scores': cv_scores,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            
            print(f"    Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    Recall: {metrics['recall_macro']:.4f}")
            print(f"    F1-Score: {metrics['f1_macro']:.4f}")
        
        # PCA Components analysis
        print(f"\n--- PCA Components ---")
        for model_name, model in models.items():
            print(f"  {model_name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_pca, y_pca, cv=kfold, scoring='accuracy')
            y_pred_cv = cross_val_predict(model, X_pca, y_pca, cv=kfold)
            
            # Get probability predictions for ROC/AUC
            y_pred_proba_cv = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba_cv = cross_val_predict(model, X_pca, y_pca, cv=kfold, method='predict_proba')
                except Exception as e:
                    print(f"    Warning: Could not get probability predictions for {model_name}: {str(e)}")
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_pca, y_pred_cv, y_pred_proba_cv)
            
            # Feature importance for Decision Tree
            model.fit(X_pca, y_pca)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
            
            results['pca_results'][model_name] = {
                'cv_scores': cv_scores,
                'mean_accuracy': cv_scores.mean(),
                'std_accuracy': cv_scores.std(),
                'metrics': metrics,
                'feature_importance': feature_importance
            }
            
            print(f"    Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            print(f"    Precision: {metrics['precision_macro']:.4f}")
            print(f"    Recall: {metrics['recall_macro']:.4f}")
            print(f"    F1-Score: {metrics['f1_macro']:.4f}")
        
        results['feature_names'] = feature_cols
        results['pca_components'] = X_pca.columns.tolist()
        
        self.results[f"{period_name}_{balance_method}"] = results
        return results

    def create_visualizations(self, save_dir="classification_plots"):
        """Create and save visualizations with metric values displayed"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Get unique balance methods
        balance_methods = set()
        for key in self.results.keys():
            if '_' in key:
                balance_method = key.split('_')[-1]
                balance_methods.add(balance_method)

        for balance_method in balance_methods:
            print(f"\nCreating visualizations for {balance_method.upper()}...")

            method_dir = f"{save_dir}/{balance_method}"
            os.makedirs(method_dir, exist_ok=True)

            try:
                self.plot_individual_model_metrics(method_dir, balance_method)
                print(f"  ✅ Individual model metrics plotted")
            except Exception as e:
                print(f"  ❌ Error plotting individual metrics: {str(e)}")

            try:
                self.plot_performance_comparison(method_dir, balance_method)
                print(f"  ✅ Performance comparison plotted")
            except Exception as e:
                print(f"  ❌ Error plotting performance: {str(e)}")

            try:
                self.plot_confusion_matrices(method_dir, balance_method)
                print(f"  ✅ Confusion matrices plotted")
            except Exception as e:
                print(f"  ❌ Error plotting confusion matrices: {str(e)}")

            try:
                self.plot_feature_importance(method_dir, balance_method)
                print(f"  ✅ Feature importance plotted")
            except Exception as e:
                print(f"  ❌ Error plotting feature importance: {str(e)}")

            try:
                self.plot_roc_comparison(method_dir, balance_method)
                print(f"  ✅ ROC comparison plotted")
            except Exception as e:
                print(f"  ❌ Error plotting ROC comparison: {str(e)}")

    def plot_individual_model_metrics(self, save_dir, balance_method):
        """Plot individual model metrics for each period and model - Original and PCA separated"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        models = ['DecisionTree', 'KNN']

        for period in periods:
            period_key = f"{period}_{balance_method}"
            if period_key not in self.results:
                continue

            # ===== 1️⃣ Original Features 차트 =====
            available_models = [m for m in models if m in self.results[period_key]['original_results']]
            if available_models:
                self._create_individual_chart(
                    available_models, 
                    self.results[period_key]['original_results'],
                    f'{save_dir}/individual_model_metrics_ORIGINAL_{period.replace(" ", "_")}_{balance_method}.png',
                    f'Original Features - {period}',
                    self.results[period_key]['balance_method_name'],
                    'steelblue'  # Original Features는 파란색 계열
                )

            # ===== 2️⃣ PCA Components 차트 =====  
            available_models = [m for m in models if m in self.results[period_key]['pca_results']]
            if available_models:
                self._create_individual_chart(
                    available_models,
                    self.results[period_key]['pca_results'], 
                    f'{save_dir}/individual_model_metrics_PCA_{period.replace(" ", "_")}_{balance_method}.png',
                    f'PCA Components - {period}',
                    self.results[period_key]['balance_method_name'],
                    'darkorange'  # PCA Components는 주황색 계열
                )

    def _create_individual_chart(self, available_models, results, save_path, title, balance_method_name, base_color):
        """Helper function to create individual model metrics chart"""
        n_models = len(available_models)
    
        if n_models == 0:
            return
        
        # 모델이 많으면 2행으로 배치
        if n_models <= 2:
            rows, cols = 1, n_models
            figsize = (7*n_models, 8)
        else:
            rows, cols = 2, (n_models + 1) // 2
            figsize = (7*cols, 8*rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif rows == 2:
            axes = axes.flatten()

        for i, model in enumerate(available_models):
            ax = axes[i]
            metrics = results[model]['metrics']

            metric_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1-Score\n(Macro)']
            metric_values = [
                metrics['accuracy'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                metrics['f1_macro']
            ]

            if metrics['roc_auc_results'] is not None:
                metric_names.append('AUC\n(Macro)')
                metric_values.append(metrics['roc_auc_results']['roc_auc']['macro'])

            # 색상 설정 - base_color 기반으로 통일감 있게
            if base_color == 'steelblue':  # Original Features
                colors = ['darkblue', 'navy', 'mediumblue', 'royalblue', 'steelblue'][:len(metric_values)]
            else:  # PCA Components (darkorange)
                colors = ['darkorange', 'orangered', 'chocolate', 'coral', 'orange'][:len(metric_values)]

            bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

            # 값 표시
            for bar, val in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

            # 모델 타입 표시
            model_type = "Base Model"
            ax.set_title(f'{model_type}: {model}', fontsize=12, fontweight='bold', pad=15)
            ax.set_ylabel('Metric Values', fontsize=11)
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=0, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)

            # 배경색 구분
            bg_color = 'lightblue' if base_color == 'steelblue' else 'lightyellow'
            ax.text(0.02, 0.98, f'Method: {balance_method_name}',
                   transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.8),
                   verticalalignment='top')

        # 빈 subplot 숨기기
        if n_models < len(axes):
            for j in range(n_models, len(axes)):
                axes[j].set_visible(False)

        # 제목에 피처 타입 표시
        feature_type = "Original Features" if base_color == 'steelblue' else "PCA Components"
        fig.suptitle(f'{title} ({feature_type})', fontsize=16, fontweight='bold', y=0.98)
    
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_performance_comparison(self, save_dir, balance_method):
        """Plot performance comparison with values displayed"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        period_keys = [f"{period}_{balance_method}" for period in periods]
        
        if not all(key in self.results for key in period_keys):
            return
            
        models = ['DecisionTree', 'KNN']
        metrics_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Original Features
        for i, metric in enumerate(metrics_names):
            ax = axes[i]
            
            x = np.arange(len(models))
            width = 0.35
            
            for j, period_key in enumerate(period_keys):
                period = periods[j]
                values = []
                
                for model in models:
                    if model in self.results[period_key]['original_results']:
                        if metric == 'accuracy':
                            val = self.results[period_key]['original_results'][model]['mean_accuracy']
                        else:
                            val = self.results[period_key]['original_results'][model]['metrics'][metric]
                        values.append(val)
                    else:
                        values.append(0)
                
                bars = ax.bar(x + j*width, values, width, label=period, alpha=0.8)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} - Original Features ({balance_method.upper()})')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison_original.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # PCA Components
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            ax = axes[i]
            
            x = np.arange(len(models))
            width = 0.35
            
            for j, period_key in enumerate(period_keys):
                period = periods[j]
                values = []
                
                for model in models:
                    if model in self.results[period_key]['pca_results']:
                        if metric == 'accuracy':
                            val = self.results[period_key]['pca_results'][model]['mean_accuracy']
                        else:
                            val = self.results[period_key]['pca_results'][model]['metrics'][metric]
                        values.append(val)
                    else:
                        values.append(0)
                
                bars = ax.bar(x + j*width, values, width, label=period, alpha=0.8)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                               f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} - PCA Components ({balance_method.upper()})')
            ax.set_xticks(x + width/2)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/performance_comparison_pca.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrices(self, save_dir, balance_method):
        """Plot confusion matrices with enhanced layout and value display"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        models = ['DecisionTree', 'KNN']

        for period in periods:
            period_key = f"{period}_{balance_method}"
            if period_key not in self.results:
                continue

            fig, axes = plt.subplots(2, 4, figsize=(18, 12))
            
            col_idx = 0
            
            # Original Features
            for model in models:
                if model in self.results[period_key]['original_results']:
                    # Confusion Matrix
                    ax_cm = axes[0, col_idx]
                    cm = self.results[period_key]['original_results'][model]['metrics']['confusion_matrix']
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                               xticklabels=['Low', 'Medium', 'High'],
                               yticklabels=['Low', 'Medium', 'High'],
                               annot_kws={'fontsize': 12, 'fontweight': 'bold'})
                    ax_cm.set_title(f'{model} (Original)\nConfusion Matrix', 
                                   fontsize=12, fontweight='bold', pad=15)
                    ax_cm.set_xlabel('Predicted', fontsize=11)
                    ax_cm.set_ylabel('Actual', fontsize=11)
                    
                    # Per-class metrics
                    ax_metrics = axes[0, col_idx + 1]
                    metrics = self.results[period_key]['original_results'][model]['metrics']
                    classes = ['Low', 'Medium', 'High']
                    
                    x = np.arange(len(classes))
                    width = 0.25
                    
                    bars1 = ax_metrics.bar(x - width, metrics['precision_per_class'], width, 
                                          label='Precision', alpha=0.8, color='skyblue')
                    bars2 = ax_metrics.bar(x, metrics['recall_per_class'], width, 
                                          label='Recall', alpha=0.8, color='lightcoral')
                    bars3 = ax_metrics.bar(x + width, metrics['f1_per_class'], width, 
                                          label='F1-Score', alpha=0.8, color='lightgreen')
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2, bars3]:
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:
                                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                              f'{height:.2f}', ha='center', va='bottom', 
                                              fontsize=9, fontweight='bold')
                    
                    ax_metrics.set_xlabel('Classes', fontsize=11)
                    ax_metrics.set_ylabel('Score', fontsize=11)
                    ax_metrics.set_title(f'{model} (Original)\nPer-Class Metrics',
                                       fontsize=12, fontweight='bold', pad=15)
                    ax_metrics.set_xticks(x)
                    ax_metrics.set_xticklabels(classes)
                    ax_metrics.legend(fontsize=10)
                    ax_metrics.set_ylim(0, 1.1)
                    ax_metrics.grid(True, alpha=0.3)
                    
                    col_idx += 2
            
            col_idx = 0
            
            # PCA Components
            for model in models:
                if model in self.results[period_key]['pca_results']:
                    # Confusion Matrix
                    ax_cm = axes[1, col_idx]
                    cm = self.results[period_key]['pca_results'][model]['metrics']['confusion_matrix']
                    
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm,
                               xticklabels=['Low', 'Medium', 'High'],
                               yticklabels=['Low', 'Medium', 'High'],
                               annot_kws={'fontsize': 12, 'fontweight': 'bold'})
                    ax_cm.set_title(f'{model} (PCA)\nConfusion Matrix',
                                   fontsize=12, fontweight='bold', pad=15)
                    ax_cm.set_xlabel('Predicted', fontsize=11)
                    ax_cm.set_ylabel('Actual', fontsize=11)
                    
                    # Per-class metrics
                    ax_metrics = axes[1, col_idx + 1]
                    metrics = self.results[period_key]['pca_results'][model]['metrics']
                    classes = ['Low', 'Medium', 'High']
                    
                    x = np.arange(len(classes))
                    width = 0.25
                    
                    bars1 = ax_metrics.bar(x - width, metrics['precision_per_class'], width, 
                                          label='Precision', alpha=0.8, color='lightblue')
                    bars2 = ax_metrics.bar(x, metrics['recall_per_class'], width, 
                                          label='Recall', alpha=0.8, color='lightpink')
                    bars3 = ax_metrics.bar(x + width, metrics['f1_per_class'], width, 
                                          label='F1-Score', alpha=0.8, color='lightseagreen')
                    
                    # Add value labels on bars
                    for bars in [bars1, bars2, bars3]:
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:
                                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                              f'{height:.2f}', ha='center', va='bottom', 
                                              fontsize=9, fontweight='bold')
                    
                    ax_metrics.set_xlabel('Classes', fontsize=11)
                    ax_metrics.set_ylabel('Score', fontsize=11)
                    ax_metrics.set_title(f'{model} (PCA)\nPer-Class Metrics',
                                       fontsize=12, fontweight='bold', pad=15)
                    ax_metrics.set_xticks(x)
                    ax_metrics.set_xticklabels(classes)
                    ax_metrics.legend(fontsize=10)
                    ax_metrics.set_ylim(0, 1.1)
                    ax_metrics.grid(True, alpha=0.3)
                    
                    col_idx += 2
            
            balance_method_name = self.results[period_key]['balance_method_name']
            fig.suptitle(f'{period} - Classification Results\n({balance_method_name})', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/confusion_matrices_{period.replace(" ", "_")}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_feature_importance(self, save_dir, balance_method):
        """Plot feature importance for Decision Tree with English names"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        period_keys = [f"{period}_{balance_method}" for period in periods]

        if not all(key in self.results for key in period_keys):
            return

        before_key = period_keys[0]
        after_key = period_keys[1]

        # Check if Decision Tree results exist
        if ('DecisionTree' not in self.results[before_key]['original_results'] or
            'DecisionTree' not in self.results[after_key]['original_results']):
            return

        before_importance = self.results[before_key]['original_results']['DecisionTree']['feature_importance']
        after_importance = self.results[after_key]['original_results']['DecisionTree']['feature_importance']

        if before_importance is None or after_importance is None:
            return

        feature_names = self.results[before_key]['feature_names']  # Already in English

        # Create comparison
        importance_data = []
        for i, feature in enumerate(feature_names):
            importance_data.append({
                'feature': feature,
                'before': before_importance[i] if i < len(before_importance) else 0,
                'after': after_importance[i] if i < len(after_importance) else 0
            })

        importance_df = pd.DataFrame(importance_data)
        importance_df['change'] = importance_df['after'] - importance_df['before']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Feature importance comparison
        x = np.arange(len(importance_df))
        width = 0.35

        bars1 = ax1.bar(x - width/2, importance_df['before'], width, 
                       label='Before Redevelopment', alpha=0.8, color='steelblue')
        bars2 = ax1.bar(x + width/2, importance_df['after'], width, 
                       label='After Redevelopment', alpha=0.8, color='darkorange')

        ax1.set_xlabel('Features', fontsize=12)
        ax1.set_ylabel('Feature Importance', fontsize=12)
        ax1.set_title(f'Decision Tree Feature Importance Comparison\n({balance_method.upper()})', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(importance_df['feature'], rotation=45, ha='right', fontsize=10)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for i, (before_val, after_val) in enumerate(zip(importance_df['before'], importance_df['after'])):
            if before_val > 0:
                ax1.text(i - width/2, before_val + 0.005, f'{before_val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            if after_val > 0:
                ax1.text(i + width/2, after_val + 0.005, f'{after_val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Importance change
        colors = ['red' if x < 0 else 'blue' for x in importance_df['change']]
        bars = ax2.bar(x, importance_df['change'], color=colors, alpha=0.7)

        ax2.set_xlabel('Features', fontsize=12)
        ax2.set_ylabel('Importance Change (After - Before)', fontsize=12)
        ax2.set_title(f'Feature Importance Change\n({balance_method.upper()})', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(importance_df['feature'], rotation=45, ha='right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, importance_df['change']):
            height = bar.get_height()
            if abs(height) > 0.001:
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.002 if height >= 0 else -0.002),
                        f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                        fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_roc_comparison(self, save_dir, balance_method):
        """Plot ROC curves comparison with AUC values displayed"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        period_keys = [f"{period}_{balance_method}" for period in periods]
        
        if not all(key in self.results for key in period_keys):
            return
            
        models = ['DecisionTree', 'KNN']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Original Features
        for i, model in enumerate(models):
            ax = axes[0, i]
            
            for j, period_key in enumerate(period_keys):
                period = periods[j]
                if (model in self.results[period_key]['original_results'] and
                    self.results[period_key]['original_results'][model]['metrics']['roc_auc_results']):
                    
                    roc_results = self.results[period_key]['original_results'][model]['metrics']['roc_auc_results']
                    
                    # Plot macro average
                    if 'macro' in roc_results['fpr']:
                        ax.plot(roc_results['fpr']['macro'], roc_results['tpr']['macro'],
                               label=f'{period} (AUC = {roc_results["roc_auc"]["macro"]:.3f})',
                               linewidth=3, linestyle='-' if j == 0 else '--')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{model} ROC Curve (Original Features)', fontsize=13, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # PCA Components
        for i, model in enumerate(models):
            ax = axes[1, i]
            
            for j, period_key in enumerate(period_keys):
                period = periods[j]
                if (model in self.results[period_key]['pca_results'] and
                    self.results[period_key]['pca_results'][model]['metrics']['roc_auc_results']):
                    
                    roc_results = self.results[period_key]['pca_results'][model]['metrics']['roc_auc_results']
                    
                    # Plot macro average
                    if 'macro' in roc_results['fpr']:
                        ax.plot(roc_results['fpr']['macro'], roc_results['tpr']['macro'],
                               label=f'{period} (AUC = {roc_results["roc_auc"]["macro"]:.3f})',
                               linewidth=3, linestyle='-' if j == 0 else '--')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title(f'{model} ROC Curve (PCA Components)', fontsize=13, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'ROC Curves Comparison ({balance_method.upper()})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/roc_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def print_results(classifier):
    """Print enhanced classification results with English feature names"""
    print(f"\n{'='*100}")
    print(f"CLASSIFICATION ANALYSIS RESULTS: ORIGINAL vs SMOTE COMPARISON")
    print(f"{'='*100}")

    # Group results by balance method
    balance_methods = {}
    for key, result in classifier.results.items():
        method = result['balance_method']
        if method not in balance_methods:
            balance_methods[method] = {}
        period = result['period']
        balance_methods[method][period] = result

    models = ['DecisionTree', 'KNN']
    method_names = {'none': 'ORIGINAL DISTRIBUTION', 'smote': 'SMOTE BALANCED'}

    for method_name, method_results in balance_methods.items():
        display_name = method_names.get(method_name, method_name.upper())
        print(f"\n{'='*100}")
        print(f"RESULTS FOR {display_name}")
        print(f"{'='*100}")

        for period, result in method_results.items():
            print(f"\n📊 {period}")
            print("=" * 90)

            data_info = result['data_info']
            balance_method_name = result['balance_method_name']
            print(f"📋 Data Information:")
            print(f"  Balance Method: {balance_method_name}")
            print(f"  Original Samples: {data_info['n_samples']}")
            print(f"  Processed Samples: {data_info['n_samples_processed']}")
            print(f"  Original Features: {data_info['n_features_original']}")
            print(f"  PCA Components: {data_info['n_features_pca']} (explained variance: {data_info['pca_variance_explained']:.3f})")

            print(f"\n🔍 Original Features Performance:")
            print(f"{'Model':<15} {'Accuracy':<10} {'CV Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
            print("-" * 85)

            for model in models:
                if model in result['original_results']:
                    metrics = result['original_results'][model]['metrics']
                    cv_scores = result['original_results'][model]['cv_scores']

                    auc_score = "N/A"
                    if metrics['roc_auc_results'] is not None:
                        auc_score = f"{metrics['roc_auc_results']['roc_auc']['macro']:.3f}"

                    print(f"{model:<15} {metrics['accuracy']:<10.3f} {cv_scores.mean():<10.3f} "
                          f"{metrics['precision_macro']:<10.3f} {metrics['recall_macro']:<10.3f} "
                          f"{metrics['f1_macro']:<10.3f} {auc_score:<10}")

            print(f"\n🔍 PCA Components Performance:")
            print(f"{'Model':<15} {'Accuracy':<10} {'CV Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
            print("-" * 85)

            for model in models:
                if model in result['pca_results']:
                    metrics = result['pca_results'][model]['metrics']
                    cv_scores = result['pca_results'][model]['cv_scores']

                    auc_score = "N/A"
                    if metrics['roc_auc_results'] is not None:
                        auc_score = f"{metrics['roc_auc_results']['roc_auc']['macro']:.3f}"

                    print(f"{model:<15} {metrics['accuracy']:<10.3f} {cv_scores.mean():<10.3f} "
                          f"{metrics['precision_macro']:<10.3f} {metrics['recall_macro']:<10.3f} "
                          f"{metrics['f1_macro']:<10.3f} {auc_score:<10}")

            # Best performing model summary
            best_model = None
            best_accuracy = 0
            for model in models:
                if model in result['original_results']:
                    acc = result['original_results'][model]['metrics']['accuracy']
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = model

            if best_model:
                print(f"\n🏆 Best Performing Model: {best_model} (Accuracy: {best_accuracy:.3f})")

                # Show per-class metrics for best model
                best_metrics = result['original_results'][best_model]['metrics']
                classes = ['Low', 'Medium', 'High']

                print(f"\n📈 {best_model} Per-Class Metrics (Original Features):")
                print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                print("-" * 55)

                for i, cls in enumerate(classes):
                    print(f"{cls:<15} {best_metrics['precision_per_class'][i]:<10.3f} "
                          f"{best_metrics['recall_per_class'][i]:<10.3f} {best_metrics['f1_per_class'][i]:<10.3f} "
                          f"{best_metrics['support_per_class'][i]:<8}")

            # Feature importance for Decision Tree if available
            if ('DecisionTree' in result['original_results'] and 
                result['original_results']['DecisionTree']['feature_importance'] is not None):
                print(f"\n🎯 Decision Tree Feature Importance (Top 5):")
                importance = result['original_results']['DecisionTree']['feature_importance']
                feature_names = result['feature_names']  # Already in English
                
                # Create feature importance pairs and sort
                feat_imp_pairs = list(zip(feature_names, importance))
                feat_imp_pairs.sort(key=lambda x: x[1], reverse=True)
                
                for i, (feat, imp) in enumerate(feat_imp_pairs[:5]):
                    print(f"  {i+1}. {feat}: {imp:.4f}")

        # Feature importance comparison between periods
        periods = list(method_results.keys())
        if len(periods) >= 2:
            print(f"\n🎯 Feature Importance Comparison ({display_name}):")
            print("-" * 90)

            before_period = 'Before Redevelopment'
            after_period = 'After Redevelopment'

            if (before_period in method_results and after_period in method_results and
                'DecisionTree' in method_results[before_period]['original_results'] and
                'DecisionTree' in method_results[after_period]['original_results']):

                before_importance = method_results[before_period]['original_results']['DecisionTree']['feature_importance']
                after_importance = method_results[after_period]['original_results']['DecisionTree']['feature_importance']

                if before_importance is not None and after_importance is not None:
                    print(f"\n📊 Decision Tree Feature Importance Changes:")
                    feature_names = method_results[before_period]['feature_names']

                    print(f"{'Feature':<25} {'Before':<10} {'After':<10} {'Change':<10} {'Status':<12}")
                    print("-" * 75)

                    for i, feature in enumerate(feature_names):
                        before_val = before_importance[i] if i < len(before_importance) else 0
                        after_val = after_importance[i] if i < len(after_importance) else 0
                        change = after_val - before_val

                        if abs(change) < 0.01:
                            status = "Stable"
                        elif change > 0:
                            status = "↗️ Increased"
                        else:
                            status = "↘️ Decreased"

                        print(f"{feature:<25} {before_val:<10.3f} {after_val:<10.3f} {change:>+10.3f} {status:<12}")

    print(f"\n{'='*100}")
    print(f"✅ Enhanced Classification Analysis Complete")
    print(f"📊 Target classes: High > Medium > Low (Elderly-Friendliness Grade)")
    print(f"📈 All feature names converted to English for better readability")
    print(f"📊 Enhanced visualizations with metric values displayed on charts")
    print(f"{'='*100}")

def run_classification_analysis():
    """Run complete enhanced classification analysis"""
    
    print("🚀 Starting Enhanced Classification Analysis: Original vs SMOTE")
    print("📊 Features: English names, enhanced visualizations with metric values")
    
    # Load data - adjust file paths as needed
    before = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/redevelopment_before_2017_2019.csv")
    after = pd.read_csv("/Users/wooyoungcho/Desktop/Univ/Data Science Term Project/Seongnam-redevelopment-elderly-study/수원시/Data(before_preprocessing)/redevelopment_after_2023.csv")
    
    print(f"Before redevelopment: {len(before)} samples")
    print(f"After redevelopment: {len(after)} samples")
    
    # Initialize classifier
    classifier = ElderlyFriendlinessClassifier(before, after)
    
    # Create save directory
    save_dir = "enhanced_classification_results"
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Run analysis for both methods
    balance_methods = ['none', 'smote']
    
    for method in balance_methods:
        print(f"\n--- {method.upper()} Method ---")
        classifier.run_classification_analysis("Before Redevelopment", before, balance_method=method)
        classifier.run_classification_analysis("After Redevelopment", after, balance_method=method)
    
    # Create enhanced visualizations
    classifier.create_visualizations(save_dir)
    
    # Print enhanced results
    print_results(classifier)
    
    print(f"\n=== Enhanced Classification Analysis Complete ===")
    print(f"Results saved in '{save_dir}' folder:")
    print("- none/: Original distribution results")
    print("- smote/: SMOTE balanced results")
    print("\nEnhanced files generated:")
    print("- individual_model_metrics_ORIGINAL_*.png: Individual model performance with values")
    print("- individual_model_metrics_PCA_*.png: PCA model performance with values")
    print("- performance_comparison_original.png: Original features performance comparison")
    print("- performance_comparison_pca.png: PCA components performance comparison")
    print("- confusion_matrices_*.png: Confusion matrices with per-class metrics and values")
    print("- feature_importance.png: Decision Tree feature importance with English names")
    print("- roc_comparison.png: ROC curves comparison with AUC values")
    print("\nModels analyzed:")
    print("1. Decision Tree Classifier with Original Features (standardized) and K-fold CV")
    print("2. Decision Tree Classifier with PCA Components and K-fold CV")
    print("3. KNN Classifier with Original Features (standardized) and K-fold CV")
    print("4. KNN Classifier with PCA Components and K-fold CV")
    print("\nEnhanced Features:")
    print("🔄 ALL FEATURES CONVERTED TO POSITIVE DIRECTION (higher = better)")
    print("🏷️ All feature names converted to English:")
    print("   - Age: Original age variable")
    print("   - Residence_Period: Years of residence in the area")
    print("   - Future_Residence_Intent: Future residence intention (higher = stronger)")
    print("   - Settlement_Mindset: Settlement mindset (higher = stronger)")
    print("   - Place_Attachment: Place attachment (higher = stronger)")
    print("   - Housing_Satisfaction: Housing satisfaction (higher = better)")
    print("   - Monthly_Income: Monthly household income (higher = better)")
    print("   - Debt_Free: 1=No debt(good), 0=Has debt(bad)")
    print("   - Life_Satisfaction: Life satisfaction (higher = better)")
    print("   - Public_Transport_Satisfaction: Public transport satisfaction (higher = better)")
    print("\n📊 Enhanced Visualizations:")
    print("   - Metric values displayed directly on all bar charts")
    print("   - Color-coded charts (Blue for Original Features, Orange for PCA)")
    print("   - Individual model performance charts separated by feature type")
    print("   - Enhanced confusion matrices with value annotations")
    print("   - Feature importance changes with directional indicators")
    print("   - ROC curves with AUC values prominently displayed")
    print("\n🎯 Target Variable: Low < Medium < High (Elderly-Friendliness Grade)")
    print("   - Low: Low elderly-friendliness")
    print("   - Medium: Medium elderly-friendliness") 
    print("   - High: High elderly-friendliness")
    print("\n📈 Performance Metrics:")
    print("   - Accuracy, Precision (Macro & Weighted), Recall (Macro & Weighted)")
    print("   - F1-Score (Macro & Weighted), AUC (Macro average)")
    print("   - Per-class metrics for detailed analysis")
    print("   - Cross-validation with 5-fold CV for robust evaluation")
    
    return classifier

if __name__ == "__main__":
    classifier = run_classification_analysis()