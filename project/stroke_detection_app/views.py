from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from django.conf import settings
import threading
import os
import time
from django.shortcuts import render
import subprocess


# Minimal safe integration: we provide a simple status page and a developer-trigger
# that launches the existing fusion.main() in a background thread. Running the
# detection uses camera/microphone resources and is intended for local/dev use.

LAST_RUN_FILE = os.path.join(os.path.dirname(__file__), 'last_result.txt')

def index(request):
    return render(request, 'stroke_detection_app/index.html')


def live(request):
    return render(request, 'stroke_detection_app/live.html')


def debug_face(request):
    """Debug endpoint: run face inference on a bundled test image and return JSON."""
    try:
        test_img = os.path.join(os.path.dirname(__file__), 'face', 'testimages', 'eyaaaaa.png')
        if not os.path.exists(test_img):
            # fallback to any image in testimages
            imgs = os.listdir(os.path.join(os.path.dirname(__file__), 'face', 'testimages'))
            if not imgs:
                return JsonResponse({'error': "Aucune image de test trouvée"}, status=500)
            test_img = os.path.join(os.path.dirname(__file__), 'face', 'testimages', imgs[0])

        from . import inference
        result = inference.run_face_on_file(test_img)
        return JsonResponse({'test_image': os.path.basename(test_img), 'result': result}, safe=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print('[stroke] debug_face error:', e)
        print(tb)
        return JsonResponse({'error': str(e), 'traceback': tb}, status=500)


def preload(request):
    """Endpoint to trigger server-side model loading (idempotent).

    The live page calls this when it loads so model weights are in memory
    before the user captures/records, reducing perceived latency.
    """
    try:
        from . import inference
        statuses = inference.preload_models()
        return JsonResponse({'status': 'ok', 'modules': statuses}, safe=True)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print('[stroke] preload error:', e)
        print(tb)
        return JsonResponse({'status': 'error', 'error': str(e), 'traceback': tb}, status=500)

def status(request):
    # show last run result if available
    try:
        with open(LAST_RUN_FILE, 'r', encoding='utf-8') as f:
            data = f.read()
    except FileNotFoundError:
        data = "Aucune exécution pour le moment."
    return HttpResponse(data, content_type='text/plain')

def _run_fusion_and_save():
    try:
        from . import fusion
        # fusion.main() prints results to console. Run it and capture minimal note.
        fusion.main()
        with open(LAST_RUN_FILE, 'w', encoding='utf-8') as f:
            f.write('fusion.main() completed. Check server console for details.')
    except Exception as e:
        with open(LAST_RUN_FILE, 'w', encoding='utf-8') as f:
            f.write(f'Error running fusion: {e}')

def start_detection(request):
    # Developer-only endpoint: starts fusion in background thread and returns immediately.
    t = threading.Thread(target=_run_fusion_and_save, daemon=True)
    t.start()
    return HttpResponse("Démarrage du dépistage d'AVC (exécution en arrière-plan).", content_type='text/plain')


def upload(request):
    """Page to upload image and/or audio for server-side inference."""
    if request.method == 'GET':
        return render(request, 'stroke_detection_app/upload.html')

    # POST: handle uploaded files — wrap to capture unexpected errors and return JSON for AJAX
    try:
        image = request.FILES.get('image')
        audio = request.FILES.get('audio')

        if not image and not audio:
            return HttpResponse('Veuillez télécharger au moins une image ou un fichier audio.', status=400)

        upload_dir = os.path.join(settings.MEDIA_ROOT, 'stroke_uploads')
        os.makedirs(upload_dir, exist_ok=True)

        saved_image_path = None
        saved_audio_path = None
        timestamp = int(time.time())

        print('[stroke] Received upload request', 'image=', bool(image), 'audio=', bool(audio))

        if image:
            image_name = f'image_{timestamp}_{image.name}'
            saved_image_path = os.path.join(upload_dir, image_name)
            with open(saved_image_path, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)
            print(f'[stroke] Saved image: {saved_image_path}')

        if audio:
            audio_name = f'audio_{timestamp}_{audio.name}'
            saved_audio_path = os.path.join(upload_dir, audio_name)
            with open(saved_audio_path, 'wb') as f:
                for chunk in audio.chunks():
                    f.write(chunk)
            print(f'[stroke] Saved audio: {saved_audio_path}')

            # Convert common browser formats to WAV for compatibility
            lower = saved_audio_path.lower()
            if lower.endswith('.webm') or lower.endswith('.ogg') or lower.endswith('.m4a'):
                wav_path = os.path.splitext(saved_audio_path)[0] + '.wav'
                try:
                    print(f'[stroke] Converting {saved_audio_path} -> {wav_path} using ffmpeg')
                    subprocess.run([
                        'ffmpeg', '-y', '-i', saved_audio_path,
                        '-ar', '16000', '-ac', '1', wav_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    saved_audio_path = wav_path
                    print(f'[stroke] Conversion complete: {saved_audio_path}')
                except Exception as e:
                        print(f'[stroke] ffmpeg conversion failed: {e}')

        # Run server-side inference
        from . import inference

        face_result = None
        speech_result = None
        fusion_result = None

        if saved_image_path:
            try:
                face_result = inference.run_face_on_file(saved_image_path)
            except Exception as e:
                print(f'[stroke] Face inference error: {e}')
                face_result = {"face_severity": None, "face_description": f"Erreur : {e}"}

        if saved_audio_path:
            try:
                speech_result = inference.run_speech_on_file(saved_audio_path)
            except Exception as e:
                print(f'[stroke] Speech inference error: {e}')
                speech_result = {"speech_label": "Erreur", "speech_probabilities": {}}

        # Always log face and speech results to server console (for privacy, web UI will only show fusion)
        print('[stroke] Face result:', face_result)
        print('[stroke] Speech result:', speech_result)

        if face_result and speech_result:
            try:
                fusion_result = inference.run_fusion(face_result, speech_result)
            except Exception as e:
                print(f'[stroke] Fusion error: {e}')
                fusion_result = {"final_risk": "Error", "explanation": str(e)}

        # Save a short summary
        try:
            with open(LAST_RUN_FILE, 'w', encoding='utf-8') as f:
                f.write(f'face={face_result}\nspeech={speech_result}\nfusion={fusion_result}')
        except Exception:
            pass

        # If AJAX request asked for JSON, return structured JSON so the page can update inline.
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or \
                'application/json' in request.headers.get('Accept', '')

        # Return only fusion result to the web client; full details remain in last_result.txt and server logs
        result_payload = {
            'fusion': fusion_result,
            'image_url': saved_image_path and (settings.MEDIA_URL + 'stroke_uploads/' + os.path.basename(saved_image_path)),
            'audio_url': saved_audio_path and (settings.MEDIA_URL + 'stroke_uploads/' + os.path.basename(saved_audio_path)),
        }

        if is_ajax:
            return JsonResponse(result_payload, safe=True)

        return render(request, 'stroke_detection_app/result.html', result_payload)
    except Exception as e:
        # Log and return JSON error for easier debugging from the live page
        import traceback
        tb = traceback.format_exc()
        print('[stroke] Unexpected error handling upload:', e)
        print(tb)
        if request.headers.get('x-requested-with') == 'XMLHttpRequest' or 'application/json' in request.headers.get('Accept', ''):
            return JsonResponse({'error': str(e), 'traceback': tb, 'message': 'Erreur interne'}, status=500)
        return HttpResponse(f'Erreur interne : {e}\n{tb}', status=500)
