import joblib
import os
import numpy as np
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

class EnsembleVerifier:
    def __init__(self):
        self.model = None

    # In EnsembleVerifier.fit():
    def _select_hard_negatives(self, X, y, ratio=2.0):
        """Keep only hard negatives to balance dataset"""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        # Train a quick classifier to find hard negatives
        quick_clf = RandomForestClassifier(n_estimators=50, max_depth=10)
        quick_clf.fit(X, y)
        
        neg_probs = quick_clf.predict_proba(X[neg_idx])[:, 1]
        # Keep negatives with highest "pothole-ness" score
        n_keep = int(len(pos_idx) * ratio)
        hard_neg_idx = neg_idx[np.argsort(-neg_probs)[:n_keep]]
        
        keep_idx = np.concatenate([pos_idx, hard_neg_idx])
        return X[keep_idx], y[keep_idx]

    def fit(self, X, y):
        # Hard negative mining first
        # X, y = self._select_hard_negatives(X, y, ratio=3.0)
        
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"    [Balance] {n_pos} Pos vs {n_neg} Neg (Ratio 1:{ratio:.1f})")
        
        # Adaptive class weighting (generalizable)
        # More imbalance = higher weight, but cap it
        pos_weight = min(ratio * 1.5, 20.0)  # Cap at 20x to avoid instability
        class_weights = {0: 1.0, 1: 10.0}
        
        # Diverse base estimators (all kept, including LR)
        clf_svm = CalibratedClassifierCV(
            SGDClassifier(
                loss='hinge', 
                penalty='elasticnet',
                alpha=0.01,
                class_weight=class_weights,
                random_state=42
            ),
            method='sigmoid',
            cv=3
        )
        
        clf_gb = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.05,
            max_depth=12,
            l2_regularization=1.0,
            class_weight=class_weights,
            early_stopping=True,
            random_state=42
        )
        
        clf_knn = KNeighborsClassifier(
            n_neighbors=7,  # Middle ground
            weights='distance',  # Weight by inverse distance
            metric='cosine'
        )
        
        clf_lr = LogisticRegression(
            solver='lbfgs',
            class_weight=class_weights,
            C=0.5,  # Moderate regularization
            max_iter=1000,
            random_state=42
        )
        
        # Add Random Forest for diversity
        clf_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
        
        ensemble = StackingClassifier(
            estimators=[
                ('svm', clf_svm),
                ('gb', clf_gb),
                ('knn', clf_knn),
                ('lr', clf_lr),
                ('rf', clf_rf)
            ],
            final_estimator=LogisticRegression(
                class_weight=class_weights,
                C=1.0,
                max_iter=1000
            ),
            cv=5,  # 5-fold CV
            n_jobs=-1
        )
        
        # Pipeline with adaptive SVD
        n_samples, n_features = X.shape
        max_safe_dim = n_samples // 5
        target_dim = min(512, n_features - 1, max_safe_dim)
        
        steps = []
        if n_features > target_dim and target_dim > 10:
            print(f"    [Dimensionality] SVD: {n_features} -> {target_dim}")
            # steps.append(('svd', TruncatedSVD(n_components=target_dim, random_state=42)))
            # steps.append(('scaler', StandardScaler()))
        
        steps.append(('ensemble', ensemble))
        
        self.model = Pipeline(steps)
        self.model.fit(X, y)
        print("    [Ensemble] Training complete.")

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return self.model.predict_proba(X)

    def save(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self, file_path)
        print(f"    [Ensemble] Saved to {file_path}")

    @classmethod
    def load(cls, file_path):
        if not os.path.exists(file_path):
            return None
        return joblib.load(file_path)