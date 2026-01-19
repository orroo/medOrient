# main.py
import numpy as np
import time

# Optional: for colored terminal output
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = ""
        YELLOW = ""
        GREEN = ""
    class Style:
        RESET_ALL = ""

# -------------------------------
# Import your face & speech scripts
# -------------------------------
from .face.test import run_face_inference   # should return dict with 'face_severity' & 'face_description'
from .speech.test_voice import run_speech_inference  # should return dict with 'speech_label' & 'speech_probabilities'

# -------------------------------
# LLM-style reasoning
# -------------------------------
def llm_reasoning(face_result, speech_result):
    # Defensive access
    severity = face_result.get("face_severity") if isinstance(face_result, dict) else None
    face_desc = face_result.get("face_description") if isinstance(face_result, dict) else None
    speech_label = None
    speech_probs = {}
    if isinstance(speech_result, dict):
        speech_label = speech_result.get("speech_label")
        speech_probs = speech_result.get("speech_probabilities") or {}

    # Normalize strings for matching
    sl = (speech_label or "").lower()

    # Extract severity as float for thresholds
    try:
        sev = float(severity) if severity is not None else 0.0
    except Exception:
        sev = 0.0

    # Extract dysarthria probability
    dys_prob = 0.0
    try:
        dys_prob = float(speech_probs.get("Dysarthric") or speech_probs.get("dysarthric") or 0.0)
    except Exception:
        dys_prob = 0.0

    # ===== Face Analysis Details =====
    if sev >= 18:
        face_detail = "L'asymétrie faciale est marquée avec un décalage manifeste des traits faciaux (sourire asymétrique, affaissement unilatéral)."
        face_risk = "élevé"
    elif sev >= 13:
        face_detail = "Une asymétrie faciale modérée est observable avec un léger décalage des traits faciaux."
        face_risk = "modéré"
    elif sev >= 8:
        face_detail = "Une asymétrie faciale légère est détectable mais discrète."
        face_risk = "léger"
    else:
        face_detail = "La symétrie faciale apparaît normale sans anomalies évidentes."
        face_risk = "minimal"

    # Use face_desc if available and substantial, otherwise use detail
    face_section = face_desc if (isinstance(face_desc, str) and len(face_desc) > 20) else face_detail

    # ===== Speech Analysis Details =====
    if dys_prob >= 0.6:
        speech_detail = f"Les troubles de la parole sont significatifs avec une articulation clairement altérée (risque dysarthrie : {dys_prob:.0%})."
        speech_risk = "élevé"
    elif dys_prob >= 0.35:
        speech_detail = f"Une légère anomalie de la parole est détectée avec une articulation partiellement affectée (risque dysarthrie : {dys_prob:.0%})."
        speech_risk = "modéré"
    else:
        speech_detail = f"La parole semble normale sans troubles notables (risque dysarthrie : {dys_prob:.0%})."
        speech_risk = "minimal"

    # ===== Triage Decision =====
    if sev >= 18 or dys_prob >= 0.6 or ("dysarth" in sl and dys_prob >= 0.2):
        risk = "HIGH"
        color = Fore.RED
        advice = "⚠️ SE RENDRE AUX URGENCES IMMÉDIATEMENT — Signes suspects détectés."
    elif sev >= 13 or dys_prob >= 0.35:
        risk = "MODERATE"
        color = Fore.YELLOW
        advice = "⚠️ RESTER VIGILANT — Consulter un médecin rapidement pour une évaluation complète."
    else:
        risk = "LOW"
        color = Fore.GREEN
        advice = "✓ AUCUNE ACTION IMMÉDIATE — Le résultat semble rassurant, pas d'urgence détectée."

    # ===== Final Explanation (3–4 lignes avec détails + conseil) =====
    explanation = (
        f"Face : {face_section}\n"
        f"Voix : {speech_detail}\n"
        f"\n{advice}"
    )

    return {"final_risk": risk, "explanation": explanation, "color": color}

# -------------------------------
# Logging helper
# -------------------------------
def log_section(title):
    print(f"\n{'='*10} {title} {'='*10}\n")


# -------------------------------
# Main
# -------------------------------
def main():
    log_section("FACE ANALYSIS")
    start_time = time.time()
    face_result = run_face_inference()
    elapsed = time.time() - start_time
    print(f"Face Severity Score : {face_result['face_severity']:.1f}")
    print(f"Face Description    : {face_result['face_description']}")
    print(f"[INFO] Face analysis completed in {elapsed:.2f}s")

    log_section("SPEECH ANALYSIS")
    start_time = time.time()
    speech_result = run_speech_inference()
    elapsed = time.time() - start_time
    print(f"Speech Result       : {speech_result['speech_label']}")
    print(f"Speech Probabilities: Control={speech_result['speech_probabilities']['Control']:.2f}, "
          f"Dysarthric={speech_result['speech_probabilities']['Dysarthric']:.2f}")

    print(f"[INFO] Speech analysis completed in {elapsed:.2f}s")

    log_section("FUSION & FINAL DECISION")
    final_result = llm_reasoning(face_result, speech_result)
    print(final_result["color"] + f"FINAL RISK LEVEL: {final_result['final_risk']}" + Style.RESET_ALL)
    print(f"Explanation       : {final_result['explanation']}")


if __name__ == "__main__":
    main()
