import torch
from pathlib import Path

def run_face_on_file(image_path: str):
    """Run face model on an image file and return standardized dict."""
    from .face.test_image import predict_image
    return predict_image(image_path)


def run_speech_on_file(audio_path: str):
    """Run speech model on an audio file and return standardized dict."""
    from .speech import test_voice as tv

    # Ensure models are loaded in tv at import time
    X = tv.extract_features(audio_path)

    with torch.no_grad():
        logits = tv.model(X)
        probs = torch.softmax(logits, dim=1)

    control_prob = float(probs[0, 0].item())
    dys_prob = float(probs[0, 1].item())

    THRESHOLD = 0.65
    if dys_prob > THRESHOLD:
        label = "Dysarthric"
        confidence = dys_prob
    elif control_prob > THRESHOLD:
        label = "Control"
        confidence = control_prob
    else:
        label = "Uncertain"
        confidence = max(control_prob, dys_prob)

    return {
        "speech_label": label,
        "speech_probabilities": {"Control": control_prob, "Dysarthric": dys_prob},
    }


def run_fusion(face_result: dict, speech_result: dict):
    """Fuse face and speech results using existing LLM-style reasoning."""
    from .fusion import llm_reasoning
    return llm_reasoning(face_result, speech_result)


def preload_models():
    """Import modules that load heavyweight models so they initialize in memory.

    Returns a dict with status for face and speech modules.
    """
    statuses = {}
    # Import face image predictor (loads Keras model at import-time)
    try:
        from .face import test_image as _face_mod
        statuses['face'] = 'loaded'
    except Exception as e:
        statuses['face'] = f'error: {e}'

    # Import speech model (loads PyTorch model at import-time)
    try:
        from .speech import test_voice as _speech_mod
        statuses['speech'] = 'loaded'
    except Exception as e:
        statuses['speech'] = f'error: {e}'

    return statuses
