import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                            precision_score, recall_score, f1_score, precision_recall_fscore_support,
                            roc_curve, auc, roc_auc_score)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✅ XGBoost successfully imported!")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    try:
        import xgboost as xgb
        XGBOOST_AVAILABLE = True
        print("✅ XGBoost successfully installed and imported!")
    except ImportError:
        XGBOOST_AVAILABLE = False
        print("❌ Failed to install XGBoost. Continuing without XGBoost...")

# Set basic matplotlib parameters
plt.rcParams['font.size'] = 10
plt.rcParams['axes.unicode_minus'] = False

class SeongnamEnsembleAnalyzer:
    def __init__(self, before_data, after_data):
        self.before_data = before_data
        self.after_data = after_data
        self.results = {}

        self.feature_info = {
            '만나이': {'type': 'continuous', 'range': 'age in years', 'direction': 'context'},
            '지역거주기간': {'type': 'continuous', 'range': 'years of residence', 'direction': 'positive'},
            '향후거주의향': {'type': 'likert', 'range': '원래 1=강하게 동의~5=강하게 반대 → 변환후 5=거주의향 강함', 'direction': 'positive'},
            '정주의식': {'type': 'special_categorical', 'range': '높을수록 정착의식 강함', 'mapping': {1: 4, 2: 2, 3: 4, 4: 1}, 'direction': 'positive'},
            '거주지소속감': {'type': 'likert', 'range': '높을수록 소속감 강함', 'direction': 'positive'},
            '거주지만족도': {'type': 'likert', 'range': '원래 1=매우만족~5=매우불만족 → 변환후 5=만족도 높음', 'direction': 'positive'},
            '월평균가구소득': {'type': 'ordinal', 'range': '1=<100만원 ~ 8=700만원+', 'direction': 'positive'},
            '부채유무': {'type': 'binary', 'range': '원래 1=있음,2=없음 → 변환후 1=무부채(좋음), 0=유부채(나쁨)', 'direction': 'positive'},
            '삶의만족도': {'type': 'likert', 'range': '원래 1=매우만족~5=매우불만족 → 변환후 5=만족도 높음', 'direction': 'positive'}
        }

    def create_elderly_friendliness_grades(self, data):
        """Create elderly friendliness grades based on residence satisfaction - POSITIVE DIRECTION"""
        satisfaction = data['거주지만족도'].copy()
        
        # 거주지만족도를 높을수록 좋게 변환 (원래: 1=매우만족~5=매우불만족)
        max_val = satisfaction.max()
        satisfaction_positive = (max_val + 1) - satisfaction  # 5=매우만족, 1=매우불만족
        
        grades = []
        q33 = satisfaction_positive.quantile(0.33)
        q67 = satisfaction_positive.quantile(0.67)

        for score in satisfaction_positive:
            if score <= q33:
                grades.append('Low')  # 만족도 낮음 = 고령친화성 낮음
            elif score <= q67:
                grades.append('Medium')
            else:
                grades.append('High')  # 만족도 높음 = 고령친화성 높음

        return pd.Series(grades), satisfaction_positive

    def balance_classes(self, X, y, method='none'):
        """Balance classes using SMOTE or keep original distribution"""
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
            method_name = "Original Distribution (No Balancing)"
        
        print(f"\nAfter {method_name}:")
        balanced_counts = pd.Series(y_balanced).value_counts()
        for cls, count in balanced_counts.items():
            print(f"  {cls}: {count}")
            
        return X_balanced, y_balanced, method_name

    def prepare_ensemble_data(self, data):
        """Data preparation for ensemble models"""
        if 'year' in data.columns:
            odd_years = [2017, 2019, 2021, 2023]
            data = data[data['year'].isin(odd_years)].copy()

        y_grades, satisfaction_scores = self.create_elderly_friendliness_grades(data)
        feature_cols = [col for col in data.columns if col not in ['거주지만족도', 'year'] and col in self.feature_info]
        X = data[feature_cols].copy()
        X_processed = self.process_features(X, data)

        return X_processed, y_grades, feature_cols, satisfaction_scores

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
                    elif col in ['향후거주의향', '거주지만족도', '삶의만족도']:
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
        print(f"   - Residence_Satisfaction: Higher = Higher residence satisfaction")
        print(f"   - Life_Satisfaction: Higher = Higher life satisfaction")
        print(f"   - Place_Attachment: Higher = Stronger place attachment")
        print(f"   - Settlement_Mindset: Higher = Stronger settlement mindset")
        print(f"   - Debt_Free: 1=No debt(good), 0=Has debt(bad)")
        print(f"   - Age, Residence_Period, Monthly_Income: Original direction maintained")
        
        return X_processed

    def create_ensemble_features(self, X):
        """Create Original Features (standardized) and PCA Components"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_scaled)
        pca_columns = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        X_pca = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)

        return X_scaled, X_pca, scaler, pca

    def calculate_detailed_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate detailed classification metrics including ROC/AUC"""
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
                    # 3클래스 예측을 2클래스로 변환
                    y_pred_proba_adjusted = y_pred_proba
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

    def run_ensemble_analysis(self, period_name, data, balance_method='none', save_dir="ensemble_plots"):
        """Run ensemble model analysis with specified balancing method including XGBoost"""
        print(f"\n=== {period_name} Ensemble Analysis ({balance_method.upper()}) ===")

        X, y, feature_cols, satisfaction_scores = self.prepare_ensemble_data(data)
        print(f"Features prepared: {len(feature_cols)}")
        print(f"Original sample size: {len(X)}")

        X_original, X_pca, scaler, pca = self.create_ensemble_features(X)
        
        y_original = y.copy()
        
        # Apply balancing method
        X_original, y, balance_method_name = self.balance_classes(X_original, y, balance_method)
        X_pca, y_pca, _ = self.balance_classes(X_pca, y_original, balance_method)

        # Label encoder for XGBoost
        le = LabelEncoder()
        le.fit(['Low', 'Medium', 'High'])

        # Base tree models (individual powerful models)
        base_tree_models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            base_tree_models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                verbosity=0,
                use_label_encoder=False
            )
            print("✅ XGBoost added to base tree models")
        else:
            print("⚠️ XGBoost not available - continuing without XGBoost")

        # Additional base models for diversity
        other_base_models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'KNeighbors': KNeighborsClassifier(n_neighbors=5),
            'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }

        # Step 1: Evaluate base models and filter by performance
        print(f"\n--- Step 1: Evaluating Base Models (Performance Filtering) ---")
        base_model_performance = {}
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name, model in base_tree_models.items():
            print(f"  Evaluating {model_name}...")
            try:
                # XGBoost용 label encoding
                if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    y_for_model = le.transform(y)
                    y_for_cv = y_for_model
                else:
                    y_for_model = y
                    y_for_cv = y
                
                cv_scores = cross_val_score(model, X_original, y_for_cv, cv=kfold, scoring='accuracy')
                mean_accuracy = cv_scores.mean()
                base_model_performance[model_name] = mean_accuracy
                
                print(f"    Accuracy: {mean_accuracy:.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                base_model_performance[model_name] = 0.0
        
        # Step 2: Filter models with performance >= 0.5
        qualified_models = {name: perf for name, perf in base_model_performance.items() if perf >= 0.5}
        excluded_models = {name: perf for name, perf in base_model_performance.items() if perf < 0.5}
        
        print(f"\n--- Step 2: Performance Filtering Results ---")
        print(f"✅ Qualified models (Accuracy >= 0.5):")
        for name, perf in qualified_models.items():
            print(f"   - {name}: {perf:.4f}")
        
        if excluded_models:
            print(f"❌ Excluded models (Accuracy < 0.5):")
            for name, perf in excluded_models.items():
                print(f"   - {name}: {perf:.4f}")
        
        # Step 3: Create voting ensemble only with qualified tree models
        voting_components = []
        final_tree_models = {}
        
        for model_name in ['RandomForest', 'GradientBoosting', 'XGBoost']:
            if model_name in qualified_models and model_name in base_tree_models:
                final_tree_models[model_name] = base_tree_models[model_name]
                voting_components.append((model_name.lower()[:3], base_tree_models[model_name]))
        
        # Add some diversity with other models (also filtered)
        for model_name, model in other_base_models.items():
            try:
                cv_scores = cross_val_score(model, X_original, y, cv=kfold, scoring='accuracy')
                mean_accuracy = cv_scores.mean()
                if mean_accuracy >= 0.5:
                    voting_components.append((model_name.lower()[:3], model))
                    print(f"   + Added {model_name} to ensemble: {mean_accuracy:.4f}")
            except:
                pass
        
        # Create final ensemble only if we have qualified models
        voting_ensemble = {}
        if len(voting_components) >= 2:
            voting_ensemble = {
                'VotingEnsemble': VotingClassifier(
                    estimators=voting_components,
                    voting='soft'
                )
            }
            print(f"\n✅ VotingEnsemble created with {len(voting_components)} qualified models")
        else:
            print(f"\n❌ Not enough qualified models for VotingEnsemble (need >= 2, got {len(voting_components)})")

        # Combine all models in logical order: Tree models → Other models → Final ensemble
        all_models = {}
        all_models.update(final_tree_models)  # Only qualified tree models
        
        # Add qualified other models
        for model_name, model in other_base_models.items():
            try:
                cv_scores = cross_val_score(model, X_original, y, cv=kfold, scoring='accuracy')
                if cv_scores.mean() >= 0.5:
                    all_models[model_name] = model
            except:
                pass
                
        all_models.update(voting_ensemble)  # Final ensemble last

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
                'pca_variance_explained': pca.explained_variance_ratio_.sum(),
                'xgboost_available': XGBOOST_AVAILABLE
            }
        }

        print(f"\n--- Original Features Ensemble Analysis ---")
        print(f"Models to analyze: {list(all_models.keys())}")
        
        for model_name, model in all_models.items():
            print(f"  Analyzing {model_name}...")

            try:
                # XGBoost용 label encoding
                if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    y_for_model = le.transform(y)
                    y_for_cv = y_for_model
                elif 'XGBoost' in str(type(model)) or (hasattr(model, 'estimators_') and any('XGBClassifier' in str(type(est)) for name, est in getattr(model, 'named_estimators_', {}).items())):
                    # VotingEnsemble에 XGBoost가 포함된 경우
                    y_for_model = y  # VotingClassifier는 원래 라벨 사용
                    y_for_cv = y
                else:
                    y_for_model = y
                    y_for_cv = y
                
                cv_scores = cross_val_score(model, X_original, y_for_cv, cv=kfold, scoring='accuracy')
                y_pred_cv = cross_val_predict(model, X_original, y_for_cv, cv=kfold)
                
                # XGBoost 결과를 원래 라벨로 변환
                if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    y_pred_cv = le.inverse_transform(y_pred_cv)
                
                y_pred_proba_cv = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba_cv = cross_val_predict(model, X_original, y_for_cv, cv=kfold, method='predict_proba')
                    except Exception as e:
                        print(f"    Warning: Could not get probability predictions for {model_name}: {str(e)}")

                detailed_metrics = self.calculate_detailed_metrics(y, y_pred_cv, y_pred_proba_cv)
                
                # 모델 피팅
                model.fit(X_original, y_for_model)

                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'named_estimators_'):
                    # VotingEnsemble의 경우
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
                    'y_pred_proba': y_pred_proba_cv,
                    'feature_importance': feature_importance,
                    'detailed_metrics': detailed_metrics
                }

                print(f"    Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                continue

        print(f"\n--- PCA Components Ensemble Analysis ---")
        for model_name, model in all_models.items():
            print(f"  Analyzing {model_name}...")

            try:
                # XGBoost용 label encoding
                if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    y_pca_for_model = le.transform(y_pca)
                    y_pca_for_cv = y_pca_for_model
                elif 'XGBoost' in str(type(model)) or (hasattr(model, 'estimators_') and any('XGBClassifier' in str(type(est)) for name, est in getattr(model, 'named_estimators_', {}).items())):
                    y_pca_for_model = y_pca
                    y_pca_for_cv = y_pca
                else:
                    y_pca_for_model = y_pca
                    y_pca_for_cv = y_pca
                
                cv_scores = cross_val_score(model, X_pca, y_pca_for_cv, cv=kfold, scoring='accuracy')
                y_pred_cv = cross_val_predict(model, X_pca, y_pca_for_cv, cv=kfold)
                
                # XGBoost 결과를 원래 라벨로 변환
                if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    y_pred_cv = le.inverse_transform(y_pred_cv)
                
                y_pred_proba_cv = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba_cv = cross_val_predict(model, X_pca, y_pca_for_cv, cv=kfold, method='predict_proba')
                    except Exception as e:
                        print(f"    Warning: Could not get probability predictions for {model_name}: {str(e)}")

                detailed_metrics = self.calculate_detailed_metrics(y_pca, y_pred_cv, y_pred_proba_cv)
                
                # 모델 피팅 (새로운 인스턴스 생성)
                if model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                    pca_model = xgb.XGBClassifier(
                        n_estimators=100, random_state=42, max_depth=6,
                        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        eval_metric='mlogloss', verbosity=0, use_label_encoder=False
                    )
                    pca_model.fit(X_pca, y_pca_for_model)
                    model = pca_model
                else:
                    model.fit(X_pca, y_pca_for_model)

                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_
                elif hasattr(model, 'named_estimators_'):
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
                    'y_true': y_pca,
                    'y_pred': y_pred_cv,
                    'y_pred_proba': y_pred_proba_cv,
                    'feature_importance': feature_importance,
                    'detailed_metrics': detailed_metrics
                }

                print(f"    Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

            except Exception as e:
                print(f"    Error with {model_name}: {str(e)}")
                continue

        results['feature_names'] = feature_cols
        results['pca_components'] = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        results['satisfaction_scores'] = satisfaction_scores

        self.results[f"{period_name}_{balance_method}"] = results
        return results

    def plot_individual_model_metrics(self, save_dir, balance_method):
        """Plot individual model metrics for each period and model including XGBoost"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        # 논리적 순서: Tree models → Other models → Final ensemble
        models = ['RandomForest', 'GradientBoosting']
        if XGBOOST_AVAILABLE:
            models.append('XGBoost')
        models.extend(['LogisticRegression', 'KNeighbors', 'DecisionTree', 'VotingEnsemble'])

        for period in periods:
            period_key = f"{period}_{balance_method}"
            if period_key not in self.results:
                continue
                
            # Original Features - 동적으로 subplot 수 조정
            available_models = [m for m in models if m in self.results[period_key]['original_results']]
            n_models = len(available_models)
            
            if n_models == 0:
                continue
                
            # 모델이 많으면 2행으로 배치
            if n_models <= 4:
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
                metrics = self.results[period_key]['original_results'][model]['detailed_metrics']
                
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
                
                # 모델 타입별 색상 구분
                if model in ['RandomForest', 'GradientBoosting', 'XGBoost']:
                    colors = ['darkblue', 'darkred', 'darkgreen', 'orange', 'purple'][:len(metric_values)]
                elif model == 'VotingEnsemble':
                    colors = ['gold', 'gold', 'gold', 'gold', 'gold'][:len(metric_values)]
                else:
                    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'plum'][:len(metric_values)]
                
                bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # 값 표시 - 위치 조정
                for bar, val in zip(bars, metric_values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                # 모델 타입 표시
                if model in ['RandomForest', 'GradientBoosting', 'XGBoost']:
                    model_type = "Tree Model"
                elif model == 'VotingEnsemble':
                    model_type = "Final Ensemble"
                else:
                    model_type = "Base Model"
                
                ax.set_title(f'{model_type}: {model}', fontsize=12, fontweight='bold', pad=15)
                ax.set_ylabel('Metric Values', fontsize=11)
                ax.set_ylim(0, 1.1)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=0, labelsize=9)
                ax.tick_params(axis='y', labelsize=9)
                
                balance_method_name = self.results[period_key]['balance_method_name']
                ax.text(0.02, 0.98, f'Method: {balance_method_name}', 
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                       verticalalignment='top')
            
            # 빈 subplot 숨기기
            if n_models < len(axes):
                for j in range(n_models, len(axes)):
                    axes[j].set_visible(False)
            
            plt.tight_layout(pad=3.0)
            plt.savefig(f'{save_dir}/individual_model_metrics_{period.replace(" ", "_")}_{balance_method}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_roc_comparison_across_periods(self, save_dir, balance_method):
        """Plot ROC comparison across different periods for all models including XGBoost"""
        # 논리적 순서: Tree models → Final ensemble
        tree_models = ['RandomForest', 'GradientBoosting']
        if XGBOOST_AVAILABLE:
            tree_models.append('XGBoost')
        ensemble_models = ['VotingEnsemble']
        
        periods = ['Before Redevelopment', 'After Redevelopment']
        
        # Tree models + Final ensemble만 ROC 비교
        models_to_plot = tree_models + ensemble_models
        
        # 동적으로 subplot 수 조정
        available_models = []
        for model in models_to_plot:
            if any(f"{period}_{balance_method}" in self.results and 
                   model in self.results[f"{period}_{balance_method}"]['original_results'] and
                   self.results[f"{period}_{balance_method}"]['original_results'][model]['detailed_metrics']['roc_auc_results'] is not None
                   for period in periods):
                available_models.append(model)
        
        n_models = len(available_models)
        if n_models == 0:
            print(f"No models with ROC data available for {balance_method}")
            return
            
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for i, model in enumerate(available_models):
            ax = axes[i]
            colors = ['blue', 'red']
            linestyles = ['-', '--']
            
            for j, (period, color, linestyle) in enumerate(zip(periods, colors, linestyles)):
                period_key = f"{period}_{balance_method}"
                
                if (period_key in self.results and 
                    model in self.results[period_key]['original_results'] and
                    self.results[period_key]['original_results'][model]['detailed_metrics']['roc_auc_results'] is not None):
                    
                    roc_results = self.results[period_key]['original_results'][model]['detailed_metrics']['roc_auc_results']
                    
                    ax.plot(roc_results['fpr']['macro'], roc_results['tpr']['macro'], 
                           color=color, lw=2, linestyle=linestyle,
                           label=f'{period} (AUC = {roc_results["roc_auc"]["macro"]:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            
            # 모델 타입별 제목
            if model in tree_models:
                model_type = "Tree Model"
            else:
                model_type = "Final Ensemble"
            
            ax.set_title(f'{model_type}: {model}\nROC Comparison', fontsize=12, fontweight='bold')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(f'{save_dir}/roc_comparison_{balance_method}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_detailed_confusion_matrices(self, save_dir, balance_method):
        """Plot detailed confusion matrices with metrics including XGBoost"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        models = ['RandomForest', 'GradientBoosting', 'VotingEnsemble']
        if XGBOOST_AVAILABLE:
            models.append('XGBoost')

        for period in periods:
            period_key = f"{period}_{balance_method}"
            if period_key not in self.results:
                continue
                
            # 동적으로 사용 가능한 모델 확인
            available_models = [m for m in models if m in self.results[period_key]['original_results']]
            n_models = len(available_models)
            
            if n_models == 0:
                continue
            
            # 더 큰 그래프 크기로 설정 (각 모델당 2개 subplot)
            fig, axes = plt.subplots(2, n_models*2, figsize=(8*n_models, 14))
            if n_models == 1:
                axes = axes.reshape(2, 2)
            
            col_idx = 0
            
            # Original Features
            for j, model in enumerate(available_models):
                if model in self.results[period_key]['original_results']:
                    # Confusion Matrix
                    ax_cm = axes[0, col_idx]
                    metrics = self.results[period_key]['original_results'][model]['detailed_metrics']
                    cm = metrics['confusion_matrix']

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                               xticklabels=['Low', 'Medium', 'High'],
                               yticklabels=['Low', 'Medium', 'High'],
                               annot_kws={'fontsize': 11})
                    ax_cm.set_title(f'{model} (Original)\nConfusion Matrix', 
                                   fontsize=12, fontweight='bold', pad=15)
                    ax_cm.set_xlabel('Predicted', fontsize=11)
                    ax_cm.set_ylabel('Actual', fontsize=11)
                    ax_cm.tick_params(axis='both', which='major', labelsize=10)
                    
                    # Per-class metrics
                    ax_metrics = axes[0, col_idx + 1]
                    classes = ['Low', 'Medium', 'High']
                    precision_vals = metrics['precision_per_class']
                    recall_vals = metrics['recall_per_class']
                    f1_vals = metrics['f1_per_class']
                    
                    x = np.arange(len(classes))
                    width = 0.25
                    
                    bars1 = ax_metrics.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color='skyblue')
                    bars2 = ax_metrics.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='lightcoral')
                    bars3 = ax_metrics.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='lightgreen')
                    
                    # 막대 위에 값 표시
                    for bars in [bars1, bars2, bars3]:
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:  # 0이 아닌 경우만 표시
                                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                              f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                    
                    ax_metrics.set_xlabel('Classes', fontsize=11)
                    ax_metrics.set_ylabel('Score', fontsize=11)
                    ax_metrics.set_title(f'{model} (Original)\nPer-Class Metrics', 
                                       fontsize=12, fontweight='bold', pad=15)
                    ax_metrics.set_xticks(x)
                    ax_metrics.set_xticklabels(classes, fontsize=10)
                    ax_metrics.legend(fontsize=9, loc='upper right')
                    ax_metrics.set_ylim(0, 1.1)
                    ax_metrics.grid(True, alpha=0.3)
                    ax_metrics.tick_params(axis='both', which='major', labelsize=9)
                    
                    col_idx += 2

            col_idx = 0
            
            # PCA Components
            for j, model in enumerate(available_models):
                if model in self.results[period_key]['pca_results']:
                    # Confusion Matrix
                    ax_cm = axes[1, col_idx]
                    metrics = self.results[period_key]['pca_results'][model]['detailed_metrics']
                    cm = metrics['confusion_matrix']

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax_cm,
                               xticklabels=['Low', 'Medium', 'High'],
                               yticklabels=['Low', 'Medium', 'High'],
                               annot_kws={'fontsize': 11})
                    ax_cm.set_title(f'{model} (PCA)\nConfusion Matrix', 
                                   fontsize=12, fontweight='bold', pad=15)
                    ax_cm.set_xlabel('Predicted', fontsize=11)
                    ax_cm.set_ylabel('Actual', fontsize=11)
                    ax_cm.tick_params(axis='both', which='major', labelsize=10)
                    
                    # Per-class metrics
                    ax_metrics = axes[1, col_idx + 1]
                    classes = ['Low', 'Medium', 'High']
                    precision_vals = metrics['precision_per_class']
                    recall_vals = metrics['recall_per_class']
                    f1_vals = metrics['f1_per_class']
                    
                    x = np.arange(len(classes))
                    width = 0.25
                    
                    bars1 = ax_metrics.bar(x - width, precision_vals, width, label='Precision', alpha=0.8, color='lightblue')
                    bars2 = ax_metrics.bar(x, recall_vals, width, label='Recall', alpha=0.8, color='lightpink')
                    bars3 = ax_metrics.bar(x + width, f1_vals, width, label='F1-Score', alpha=0.8, color='lightseagreen')
                    
                    # 막대 위에 값 표시
                    for bars in [bars1, bars2, bars3]:
                        for bar in bars:
                            height = bar.get_height()
                            if height > 0:  # 0이 아닌 경우만 표시
                                ax_metrics.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                              f'{height:.2f}', ha='center', va='bottom', fontsize=9)
                    
                    ax_metrics.set_xlabel('Classes', fontsize=11)
                    ax_metrics.set_ylabel('Score', fontsize=11)
                    ax_metrics.set_title(f'{model} (PCA)\nPer-Class Metrics', 
                                       fontsize=12, fontweight='bold', pad=15)
                    ax_metrics.set_xticks(x)
                    ax_metrics.set_xticklabels(classes, fontsize=10)
                    ax_metrics.legend(fontsize=9, loc='upper right')
                    ax_metrics.set_ylim(0, 1.1)
                    ax_metrics.grid(True, alpha=0.3)
                    ax_metrics.tick_params(axis='both', which='major', labelsize=9)
                    
                    col_idx += 2

            balance_method_name = self.results[period_key]['balance_method_name']
            fig.suptitle(f'{period} - Model Performance\n({balance_method_name})', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # 여백 조정
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.subplots_adjust(hspace=0.3, wspace=0.25)
            
            plt.savefig(f'{save_dir}/confusion_matrices_{period.replace(" ", "_")}_{balance_method}.png', 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

    def plot_ensemble_feature_importance(self, save_dir, balance_method):
        """Plot feature importance for ensemble models including XGBoost"""
        periods = ['Before Redevelopment', 'After Redevelopment']
        period_keys = [f"{period}_{balance_method}" for period in periods]
        
        if not all(key in self.results for key in period_keys):
            return

        before_key = period_keys[0]
        after_key = period_keys[1]

        feature_mapping = {
            '만나이': 'Age',
            '지역거주기간': 'Residence_Period',
            '향후거주의향': 'Future_Residence_Intent_Pos',
            '정주의식': 'Settlement_Mindset',
            '거주지소속감': 'Place_Attachment',
            '월평균가구소득': 'Monthly_Income',
            '부채유무': 'Debt_Free',
            '삶의만족도': 'Life_Satisfaction_Pos'
        }

        # Create subplot for each model with feature importance
        models_with_importance = ['RandomForest', 'GradientBoosting']
        if XGBOOST_AVAILABLE:
            models_with_importance.append('XGBoost')

        # Filter available models
        available_models = []
        for model_name in models_with_importance:
            if (model_name in self.results[before_key]['original_results'] and 
                model_name in self.results[after_key]['original_results'] and
                self.results[before_key]['original_results'][model_name]['feature_importance'] is not None and
                self.results[after_key]['original_results'][model_name]['feature_importance'] is not None):
                available_models.append(model_name)

        if not available_models:
            print(f"No models with feature importance available for {balance_method}")
            return

        fig, axes = plt.subplots(len(available_models), 2, figsize=(16, 7*len(available_models)))
        if len(available_models) == 1:
            axes = axes.reshape(1, -1)

        for model_idx, model_name in enumerate(available_models):
            try:
                before_importance = self.results[before_key]['original_results'][model_name]['feature_importance']
                after_importance = self.results[after_key]['original_results'][model_name]['feature_importance']

                feature_names = self.results[before_key]['feature_names']

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

                ax1 = axes[model_idx, 0]
                ax2 = axes[model_idx, 1]

                x = np.arange(len(importance_df))
                width = 0.35

                bars1 = ax1.bar(x - width/2, importance_df['before'], width, label='Before Redevelopment', alpha=0.8, color='steelblue')
                bars2 = ax1.bar(x + width/2, importance_df['after'], width, label='After Redevelopment', alpha=0.8, color='darkorange')

                ax1.set_xlabel('Features', fontsize=12)
                ax1.set_ylabel('Feature Importance', fontsize=12)
                ax1.set_title(f'{model_name} Feature Importance Comparison\n({balance_method.upper()})', fontsize=13, fontweight='bold')
                ax1.set_xticks(x)
                ax1.set_xticklabels(importance_df['feature'], rotation=45, ha='right', fontsize=10)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(axis='y', labelsize=10)

                # 값 표시
                for i, (before_val, after_val) in enumerate(zip(importance_df['before'], importance_df['after'])):
                    if before_val > 0:
                        ax1.text(i - width/2, before_val + 0.005, f'{before_val:.3f}', 
                                ha='center', va='bottom', fontsize=8)
                    if after_val > 0:
                        ax1.text(i + width/2, after_val + 0.005, f'{after_val:.3f}', 
                                ha='center', va='bottom', fontsize=8)

                colors = ['red' if x < 0 else 'blue' for x in importance_df['change']]
                bars = ax2.bar(x, importance_df['change'], color=colors, alpha=0.7)

                ax2.set_xlabel('Features', fontsize=12)
                ax2.set_ylabel('Importance Change (After - Before)', fontsize=12)
                ax2.set_title(f'{model_name} Feature Importance Change\n({balance_method.upper()})', fontsize=13, fontweight='bold')
                ax2.set_xticks(x)
                ax2.set_xticklabels(importance_df['feature'], rotation=45, ha='right', fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.tick_params(axis='y', labelsize=10)

                for bar, val in zip(bars, importance_df['change']):
                    height = bar.get_height()
                    if abs(height) > 0.001:  # 작은 값은 표시하지 않음
                        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.002 if height >= 0 else -0.002),
                                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)

            except Exception as e:
                print(f"Error creating feature importance plot for {model_name}: {str(e)}")
                continue

        plt.tight_layout(pad=3.0)
        plt.savefig(f'{save_dir}/feature_importance_{balance_method}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_ensemble_visualizations(self, save_dir="ensemble_plots"):
        """Create and save ensemble visualizations for all balance methods"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Get unique balance methods from results
        balance_methods = set()
        for key in self.results.keys():
            if '_' in key:
                balance_method = key.split('_')[-1]
                balance_methods.add(balance_method)

        for balance_method in balance_methods:
            print(f"\nCreating visualizations for {balance_method.upper()} method...")
            
            # Create method-specific subdirectory
            method_dir = f"{save_dir}/{balance_method}"
            os.makedirs(method_dir, exist_ok=True)
            
            try:
                self.plot_individual_model_metrics(method_dir, balance_method)
                print(f"  ✅ Individual model metrics plotted")
            except Exception as e:
                print(f"  ❌ Error plotting individual metrics: {str(e)}")
            
            try:
                self.plot_roc_comparison_across_periods(method_dir, balance_method)
                print(f"  ✅ ROC comparison plotted")
            except Exception as e:
                print(f"  ❌ Error plotting ROC comparison: {str(e)}")
            
            try:
                self.plot_detailed_confusion_matrices(method_dir, balance_method)
                print(f"  ✅ Confusion matrices plotted")
            except Exception as e:
                print(f"  ❌ Error plotting confusion matrices: {str(e)}")
            
            try:
                self.plot_ensemble_feature_importance(method_dir, balance_method)
                print(f"  ✅ Feature importance plotted")
            except Exception as e:
                print(f"  ❌ Error plotting feature importance: {str(e)}")

        print(f"\nAll ensemble visualizations saved to {save_dir} folder!")


def print_ensemble_results(analyzer):
    """Print detailed results for ensemble analysis comparing Original vs SMOTE"""
    print(f"\n{'='*100}")
    print(f"DETAILED ENSEMBLE ANALYSIS RESULTS: ORIGINAL vs SMOTE COMPARISON")
    print(f"{'='*100}")
    
    # Group results by balance method
    balance_methods = {}
    for key, result in analyzer.results.items():
        method = result['balance_method']
        if method not in balance_methods:
            balance_methods[method] = {}
        period = result['period']
        balance_methods[method][period] = result

    models = ['RandomForest', 'GradientBoosting']
    if XGBOOST_AVAILABLE:
        models.extend(['XGBoost'])
    models.extend(['LogisticRegression', 'KNeighbors', 'DecisionTree', 'VotingEnsemble'])
    
    method_names = {
        'none': 'ORIGINAL DISTRIBUTION',
        'smote': 'SMOTE BALANCED'
    }
    
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
            print(f"  XGBoost Available: {'✅ Yes' if data_info['xgboost_available'] else '❌ No'}")
            
            print(f"\n🔍 Original Features Performance:")
            print(f"{'Model':<20} {'Accuracy':<10} {'CV Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
            print("-" * 90)
            
            for model in models:
                if model in result['original_results']:
                    metrics = result['original_results'][model]['detailed_metrics']
                    cv_scores = result['original_results'][model]['cv_scores']
                    
                    auc_score = "N/A"
                    if metrics['roc_auc_results'] is not None:
                        auc_score = f"{metrics['roc_auc_results']['roc_auc']['macro']:.3f}"
                    
                    print(f"{model:<20} {metrics['accuracy']:<10.3f} {cv_scores.mean():<10.3f} "
                          f"{metrics['precision_macro']:<10.3f} {metrics['recall_macro']:<10.3f} "
                          f"{metrics['f1_macro']:<10.3f} {auc_score:<10}")
            
            print(f"\n🔍 PCA Components Performance:")
            print(f"{'Model':<20} {'Accuracy':<10} {'CV Acc':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
            print("-" * 90)
            
            for model in models:
                if model in result['pca_results']:
                    metrics = result['pca_results'][model]['detailed_metrics']
                    cv_scores = result['pca_results'][model]['cv_scores']
                    
                    auc_score = "N/A"
                    if metrics['roc_auc_results'] is not None:
                        auc_score = f"{metrics['roc_auc_results']['roc_auc']['macro']:.3f}"
                    
                    print(f"{model:<20} {metrics['accuracy']:<10.3f} {cv_scores.mean():<10.3f} "
                          f"{metrics['precision_macro']:<10.3f} {metrics['recall_macro']:<10.3f} "
                          f"{metrics['f1_macro']:<10.3f} {auc_score:<10}")
            
            # Best performing model summary
            best_model = None
            best_accuracy = 0
            for model in models:
                if model in result['original_results']:
                    acc = result['original_results'][model]['detailed_metrics']['accuracy']
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_model = model
            
            if best_model:
                print(f"\n🏆 Best Performing Model: {best_model} (Accuracy: {best_accuracy:.3f})")
                
                # Show per-class metrics for best model
                best_metrics = result['original_results'][best_model]['detailed_metrics']
                classes = ['Low', 'Medium', 'High']
                
                print(f"\n📈 {best_model} Per-Class Metrics (Original Features):")
                print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
                print("-" * 55)
                
                for i, cls in enumerate(classes):
                    print(f"{cls:<15} {best_metrics['precision_per_class'][i]:<10.3f} "
                          f"{best_metrics['recall_per_class'][i]:<10.3f} {best_metrics['f1_per_class'][i]:<10.3f} "
                          f"{best_metrics['support_per_class'][i]:<8}")

        # Feature importance comparison within this method
        periods = list(method_results.keys())
        if len(periods) >= 2:
            print(f"\n🎯 Feature Importance Comparison ({display_name}):")
            print("-" * 90)
            
            before_period = 'Before Redevelopment'
            after_period = 'After Redevelopment'
            
            if before_period in method_results and after_period in method_results:
                # Show comparison for all models with feature importance
                feature_models = ['RandomForest', 'GradientBoosting']
                if XGBOOST_AVAILABLE:
                    feature_models.append('XGBoost')
                
                for model_name in feature_models:
                    if (model_name in method_results[before_period]['original_results'] and 
                        model_name in method_results[after_period]['original_results']):
                        
                        before_importance = method_results[before_period]['original_results'][model_name]['feature_importance']
                        after_importance = method_results[after_period]['original_results'][model_name]['feature_importance']
                        
                        if before_importance is not None and after_importance is not None:
                            print(f"\n📊 {model_name} Feature Importance:")
                            feature_names = method_results[before_period]['feature_names']
                            
                            feature_mapping = {
                                '만나이': 'Age',
                                '지역거주기간': 'Residence_Period',
                                '향후거주의향': 'Future_Residence_Intent_Pos',
                                '정주의식': 'Settlement_Mindset',
                                '거주지소속감': 'Place_Attachment',
                                '월평균가구소득': 'Monthly_Income',
                                '부채유무': 'Debt_Free',
                                '삶의만족도': 'Life_Satisfaction_Pos'
                            }
                            
                            print(f"{'Feature':<25} {'Before':<10} {'After':<10} {'Change':<10} {'Status':<12}")
                            print("-" * 75)
                            
                            for i, feature in enumerate(feature_names):
                                eng_name = feature_mapping.get(feature, feature)
                                before_val = before_importance[i] if i < len(before_importance) else 0
                                after_val = after_importance[i] if i < len(after_importance) else 0
                                change = after_val - before_val
                                
                                if abs(change) < 0.01:
                                    status = "Stable"
                                elif change > 0:
                                    status = "↗️ Increased"
                                else:
                                    status = "↘️ Decreased"
                                
                                print(f"{eng_name:<25} {before_val:<10.3f} {after_val:<10.3f} {change:>+10.3f} {status:<12}")

    print(f"\n{'='*100}")
    print(f"✅ Enhanced Analysis Complete - Original vs SMOTE Comparison")
    if XGBOOST_AVAILABLE:
        print(f"🚀 XGBoost successfully integrated into ensemble analysis")
    else:
        print(f"⚠️ XGBoost was not available - analysis completed with other models")
    print(f"📊 Comparison shows impact of SMOTE balancing on model performance")
    print(f"🎯 Performance filtering ensures only reliable models (Accuracy >= 0.5) in ensemble")
    print(f"{'='*100}")


def run_complete_ensemble_analysis():
    """Run complete ensemble analysis comparing Original vs SMOTE balanced data"""

    print("🚀 Starting Enhanced Ensemble Analysis: Original vs SMOTE Comparison")
    print("📁 Make sure your Google Drive is mounted!")

    before = pd.read_csv("/content/drive/MyDrive/2025-1학기 데이터과학 1조/Data/성남시 사회조사/최종 파일/redevelopment_before_2017_2019.csv")
    after = pd.read_csv("/content/drive/MyDrive/2025-1학기 데이터과학 1조/Data/성남시 사회조사/최종 파일/redevelopment_after_2023.csv")

    print(f"Before redevelopment (65+): {len(before)} samples")
    print(f"After redevelopment (65+): {len(after)} samples")

    analyzer = SeongnamEnsembleAnalyzer(before, after)

    save_dir = "seongnam_ensemble_results_original_vs_smote"
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Compare only Original vs SMOTE
    balance_methods = ['none', 'smote']
    method_names = {
        'none': 'Original Distribution',
        'smote': 'SMOTE Balanced'
    }
    
    print(f"\n=== Comparing Original vs SMOTE Balanced Data ===")
    if XGBOOST_AVAILABLE:
        print("📊 Models included: RandomForest, GradientBoosting, XGBoost, LogisticRegression, KNeighbors, DecisionTree, VotingEnsemble")
    else:
        print("📊 Models included: RandomForest, GradientBoosting, LogisticRegression, KNeighbors, DecisionTree, VotingEnsemble (XGBoost not available)")
    
    for method in balance_methods:
        method_display = method_names[method]
        print(f"\n--- Testing {method_display} ---")
        
        # Run analysis with specific balance method
        analyzer.run_ensemble_analysis("Before Redevelopment", before, balance_method=method, save_dir=save_dir)
        analyzer.run_ensemble_analysis("After Redevelopment", after, balance_method=method, save_dir=save_dir)

    # Create visualizations for all methods
    analyzer.create_ensemble_visualizations(save_dir)

    # Print detailed results
    print_ensemble_results(analyzer)

    print(f"\n=== Enhanced Ensemble Analysis Complete ===")
    print(f"Results saved in '{save_dir}' folder with subfolders for each method:")
    print("- none/: Original distribution results")
    print("- smote/: SMOTE balanced results")
    print("\nEach subfolder contains:")
    print("- individual_model_metrics_*.png: Individual model performance charts")
    print("- roc_comparison_*.png: ROC comparison across periods")
    print("- confusion_matrices_*.png: Confusion matrices with per-class metrics")
    print("- feature_importance_*.png: Feature importance comparison")
    print("\nEnhanced Ensemble Models Analyzed (in logical order):")
    print("🌳 Tree-based Models:")
    print("- Random Forest (Original & PCA)")
    print("- Gradient Boosting (Original & PCA)")
    if XGBOOST_AVAILABLE:
        print("- XGBoost (Original & PCA) ✅")
    else:
        print("- XGBoost (Not Available) ❌")
    print("\n📊 Additional Base Models:")
    print("- Logistic Regression (Original & PCA)")
    print("- K-Nearest Neighbors (Original & PCA)")
    print("- Decision Tree (Original & PCA)")
    print("\n🏆 Final Ensemble:")
    voting_components = "RandomForest + GradientBoosting"
    if XGBOOST_AVAILABLE:
        voting_components += " + XGBoost"
    voting_components += " + LogisticRegression + KNN (qualified models only)"
    print(f"- Voting Ensemble: {voting_components} (Original & PCA)")
    print("\nKey Features:")
    print("- 🔄 ALL FEATURES CONVERTED TO POSITIVE DIRECTION (higher = better)")
    print("- 📊 Original vs SMOTE comparison (no manual sampling)")
    print("- 🎯 Performance filtering: Only models with Accuracy >= 0.5 in ensemble")
    print("- Enhanced model set with XGBoost integration")
    print("- Detailed metrics: Accuracy, Precision, Recall, F1-Score, AUC")
    print("- ROC curves with before/after comparison")
    print("- Per-class performance analysis")
    print("- Feature importance comparison across all tree-based models")
    print("- Cross-validation with 5-fold CV")
    print("- Automatic model availability detection")
    print("- English-only labels for clean visualization")
    print("- Error handling for robust analysis")
    
    print("\n📊 Feature Interpretation Guide (All Positive Direction):")
    print("- Future_Residence_Intent_Pos: Higher = Stronger residence intention")
    print("- Residence_Satisfaction: Higher = Higher residence satisfaction") 
    print("- Life_Satisfaction_Pos: Higher = Higher life satisfaction")
    print("- Place_Attachment: Higher = Stronger place attachment")
    print("- Settlement_Mindset: Higher = Stronger settlement mindset")
    print("- Debt_Free: 1=No debt(good), 0=Has debt(bad)")
    print("- Residence_Period: Higher = Longer residence")
    print("- Monthly_Income: Higher = Higher income")
    print("- Age: Context dependent")
    print("\n🎯 Target Variable: Low < Medium < High (Elderly-Friendliness Grade)")
    print("- Low: Low elderly-friendliness")
    print("- Medium: Medium elderly-friendliness") 
    print("- High: High elderly-friendliness")
    print("\n🔧 Performance Filtering: Only models with Accuracy >= 0.5 included in ensemble")

    return analyzer

# 실행
analyzer = run_complete_ensemble_analysis()