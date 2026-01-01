import os
from django.shortcuts import render
from django.conf import settings
from .ml import compute_day_error

# Tune this threshold (µ+3σ from baseline) per patient/side
THRESHOLD = 7.571944129128372


def upload_view(request):
    if request.method == 'POST':
        wav = request.FILES['wav']
        save_path = settings.MEDIA_ROOT / wav.name
        with open(save_path, 'wb+') as f:
            for chunk in wav.chunks():
                f.write(chunk)

        score = compute_day_error(str(save_path))

        if score is None:
            message = 'Audio too short to evaluate.'
        else:
            if score > THRESHOLD:
                status = 'WORSE'
                color = 'red'
                note = '🔴 Possible increased turbulence (stenosis).'
            elif score < 3.52123570646443:
                status = 'BETTER'
                color = 'orange'
                note = '🟡 Unusually smooth—monitor for low flow.'
            else:
                status = 'OK'
                color = 'green'
                note = '🟢 Within normal variability.'

            message = f"<span style='color:{color}'>{note}</span>"


        # Remove file after processing if desired:
        os.remove(save_path)

        return render(request, 'vascular.html', {'message': message})

    return render(request, 'uploadVae.html')