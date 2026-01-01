from django.shortcuts import render

import os
from scipy.io import loadmat
import numpy as np
from django.conf import settings

from django.http import JsonResponse
# from .ml_model import prediction_model
import numpy as np
import torch.nn as nn
import torch
import joblib
import requests


class PatchTST(nn.Module):
    def __init__(self, input_channels=2, seq_len=125, patch_len=25, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super(PatchTST, self).__init__()
        self.patch_len = patch_len
        self.seq_len = seq_len
        self.n_patches = seq_len // patch_len

        self.embedding = nn.Linear(input_channels * patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * self.n_patches, 128)
        self.fc1 = nn.Linear(128, 2)

    def forward(self, x):
        # x: (batch_size, seq_len, input_channels)
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, self.n_patches, -1)  # (batch_size, n_patches, input_channels * patch_len)
        x = self.embedding(x)  # (batch_size, n_patches, d_model)
        x = x.permute(1, 0, 2)  # (n_patches, batch_size, d_model)
        x = self.transformer(x)
        x = x.permute(1, 0, 2).contiguous().view(batch_size, -1)
        out = self.fc(x)
        out = self.fc1(out)
        return out
    



F_D_model = PatchTST()  # rebuild the model architecture exactly
F_D_model.load_state_dict(torch.load('check_app\pkl\PatchTSL_model_weights.pt'))
F_D_model.eval()

F_D_scalerX =  joblib.load("check_app\pkl\scalerX.pkl")
F_D_scalerY =  joblib.load("check_app\pkl\scalerY.pkl")


from django.shortcuts import render, redirect
from scipy.io import loadmat
from .forms import MatFileUploadForm
import os
from django.conf import settings
import numpy as np


def load(upload_path):
    mat_data = loadmat(upload_path)
    X = mat_data['X'].copy()
    num_rows = X.shape[0]
    random_idx = np.random.randint(0, num_rows)
    return   np.expand_dims(X[random_idx], axis=0) 


def upload_mat(request):
    if request.method == 'POST':
        form = MatFileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save uploaded file
            mat_file = request.FILES['mat_file']
            upload_path = os.path.join(settings.MEDIA_ROOT, 'uploads', mat_file.name)
            os.makedirs(os.path.dirname(upload_path), exist_ok=True)
            
            with open(upload_path, 'wb+') as f:
                for chunk in mat_file.chunks():
                    f.write(chunk)
            
            # Process .mat file
            X=load(upload_path)

            X = F_D_scalerX.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X = torch.from_numpy(X).float()
            predictions = F_D_model(X)
            output = F_D_scalerY.inverse_transform(predictions.detach().numpy())
            print(output)
            # Pass results to template
            return render(request, 'result.html', {
                'prediction_sdp': output[0][0],
                'prediction_dbp': output[0][1],
                'data_sample': X.tolist()  # Show first row as example
            })
    else:
        form = MatFileUploadForm()
    
    return render(request, 'bp.html', {'form': form})

live_url = 'http://127.0.0.1:5000/mat_data'


def live_data(request):
    # response = requests.get(live_url)
    
    # if response.status_code == 200:
    #     X = np.array(response.json())  # This is your random sample as a Python dict
    #     print(X)
    # else:
    #     print(f"Error: {response.status_code}")

    # X = F_D_scalerX.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    # X = torch.from_numpy(X).float()
    # predictions = F_D_model(X)
    # output = F_D_scalerY.inverse_transform(predictions.detach().numpy())
    # print(output)
    # # Pass results to template
    # return render(request, 'result.html', {
    #     'prediction_sdp': output[0][0],
    #     'prediction_dbp': output[0][1],
    #     'data_sample': X.tolist()  # Show first row as example
    # })
    
    return render(request, 'mon.html')



