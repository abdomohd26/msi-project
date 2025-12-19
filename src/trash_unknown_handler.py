import numpy as np
from src.config import UNKNOWN_THRESHOLD, TRASH_THRESHOLD

def handle_trash_unknown(model, feature_vector, trash_class=5):
    """
    Returns class prediction with:
    - Trash confidence boost
    - Unknown handling
    """
    probs = model.predict_proba([feature_vector])[0]
    max_prob = np.max(probs)
    predicted_class = np.argmax(probs)

    if max_prob < UNKNOWN_THRESHOLD:
        return 6

    trash_prob = probs[trash_class]
    if trash_prob > TRASH_THRESHOLD and trash_prob >= 0.8 * max_prob:
        return trash_class

    return predicted_class
