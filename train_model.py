"""
Train Voice Gender ML model from voice.csv dataset
Uses XGBoost-style gradient boosting via scikit-learn
"""
import csv
import numpy as np
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Load dataset
print("Loading voice.csv dataset...")
rows = []
with open('voice.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

FEATURES = ['meanfreq','sd','median','Q25','Q75','IQR','skew','kurt',
            'sp.ent','sfm','mode','centroid','meanfun','minfun','maxfun',
            'meandom','mindom','maxdom','dfrange','modindx']

X = np.array([[float(r[f]) for f in FEATURES] for r in rows])
y = np.array([1 if r['label'] == 'male' else 0 for r in rows])

print(f"Dataset: {len(X)} samples, {len(FEATURES)} features")
print(f"Male: {sum(y)}, Female: {len(y)-sum(y)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build stacked ensemble (SVM + GBM + RF)
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, C=10, gamma='scale'))
])

gbm = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# Voting ensemble
print("Training models...")
svm_pipe.fit(X_train, y_train)
gbm.fit(X_train, y_train)
rf.fit(X_train, y_train)

svm_acc = accuracy_score(y_test, svm_pipe.predict(X_test))
gbm_acc = accuracy_score(y_test, gbm.predict(X_test))
rf_acc  = accuracy_score(y_test, rf.predict(X_test))

print(f"SVM accuracy:           {svm_acc:.4f} ({svm_acc*100:.1f}%)")
print(f"GBM accuracy:           {gbm_acc:.4f} ({gbm_acc*100:.1f}%)")
print(f"Random Forest accuracy: {rf_acc:.4f}  ({rf_acc*100:.1f}%)")

# Save all models + scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
joblib.dump(svm_pipe,  'models/svm_model.pkl')
joblib.dump(gbm,       'models/gbm_model.pkl')
joblib.dump(rf,        'models/rf_model.pkl')
joblib.dump(scaler,    'models/scaler.pkl')
joblib.dump(FEATURES,  'models/features.pkl')

print("\nAll models saved to models/ directory!")
print(f"Best model: {'GBM' if gbm_acc >= rf_acc else 'RF'} with {max(gbm_acc,rf_acc)*100:.1f}% accuracy")
