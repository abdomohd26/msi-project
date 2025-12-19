import numpy as np
from src.config import UNKNOWN_THRESHOLD

def handle_unknown(model, feature_vector):
    probs = model.predict_proba([feature_vector])[0]
    
    max_prob = np.max(probs)
    predicted_class = np.argmax(probs)

    if max_prob < UNKNOWN_THRESHOLD:
        return 6
    return predicted_class
