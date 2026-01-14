from sklearn.ensemble import RandomForestClassifier
import numpy as np

model = RandomForestClassifier(n_estimators=100)

# Dummy training data (for hackathon demo)
X_train = np.array([
    [2, 0.5, 0.8, 10],
    [6, 1.2, 0.4, 25],
    [3, 0.7, 0.7, 12],
    [8, 1.5, 0.3, 30]
])

y_train = [0, 1, 0, 1]  # 0 = Low Risk, 1 = High Risk

model.fit(X_train, y_train)

def predict_risk(features):
    feature_vector = np.array([[
        features["pause_count"],
        features["speech_rate"],
        features["vocab_richness"],
        features["avg_sentence_length"]
    ]])
    
    prob = model.predict_proba(feature_vector)[0][1]
    return prob
