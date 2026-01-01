from django import forms

class MatFileUploadForm(forms.Form):
    mat_file = forms.FileField(
        label="Upload .mat File",
        widget=forms.FileInput(attrs={
            'class': 'form-control custom-file-input',
            'accept': '.mat',
            'required': True
        }),
        help_text="Upload a MATLAB .mat file containing patient data"
    )
    
