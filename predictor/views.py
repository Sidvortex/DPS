import json
from django.shortcuts import render
from .forms  import DiabetesForm, HeartDiseaseForm
from .models import PredictionRecord
from . import ml_service

def index(request):
    disease = request.POST.get('disease', request.GET.get('disease', 'diabetes'))
    if disease not in ('diabetes', 'heart'):
        disease = 'diabetes'

    diab_form  = DiabetesForm()
    heart_form = HeartDiseaseForm()
    result = None

    if request.method == 'POST':
        if disease == 'diabetes':
            # For male patients, pregnancies field is hidden — ensure it's 0
            post_data = request.POST.copy()
            if not post_data.get('pregnancies'):
                post_data['pregnancies'] = '0'
            diab_form = DiabetesForm(post_data)
            if diab_form.is_valid():
                d = diab_form.cleaned_data
                result = ml_service.predict_diabetes(d)
                PredictionRecord.objects.create(
                    disease='diabetes', prediction=result['label'],
                    confidence=result['confidence'], risk_level=result['risk_level'],
                    age=d['age'], pregnancies=d['pregnancies'], glucose=d['glucose'],
                    blood_pressure=d['blood_pressure'], skin_thickness=d['skin_thickness'],
                    insulin=d['insulin'], bmi=d['bmi'], dpf=d['dpf'],
                )
        else:
            heart_form = HeartDiseaseForm(request.POST)
            if heart_form.is_valid():
                d = {k: float(v) if '.' in str(v) else int(v)
                     for k, v in heart_form.cleaned_data.items()}
                result = ml_service.predict_heart(d)
                PredictionRecord.objects.create(
                    disease='heart', prediction=result['label'],
                    confidence=result['confidence'], risk_level=result['risk_level'],
                    age=d['age'], sex=d['sex'], cp=d['cp'], trestbps=d['trestbps'],
                    chol=d['chol'], fbs=d['fbs'], restecg=d['restecg'],
                    thalach=d['thalach'], exang=d['exang'], oldpeak=d['oldpeak'],
                    slope=d['slope'], ca=d['ca'], thal=d['thal'],
                )

    d_high, d_low = ml_service.get_diabetes_profiles()
    h_high, h_low = ml_service.get_heart_profiles()
    chart_data    = ml_service.get_diab_chart_data()

    return render(request, 'index.html', {
        'disease':       disease,
        'diab_form':     diab_form,
        'heart_form':    heart_form,
        'result':        result,
        'diab_stats':    ml_service.get_diabetes_stats(),
        'heart_stats':   ml_service.get_heart_stats(),
        'diab_high':     d_high, 'diab_low':  d_low,
        'heart_high':    h_high, 'heart_low': h_low,
        'history':       PredictionRecord.objects.filter(disease=disease)[:8],
        'chart_json':    json.dumps(chart_data),
        'radar_json':    json.dumps(result.get('radar') if result else None),
        'fi_json':       json.dumps(result.get('feature_importance') if result else None),
        'indicators':      json.dumps(result.get('indicators') if result else {}),   # JSON string for JS
        'indicators_dict': result.get('indicators') if result else {},                # dict for Django template
        'diab_model':    ml_service.DIAB_MODEL_NAME, 'diab_acc': ml_service.DIAB_ACCURACY,
        'heart_model':   ml_service.HEART_MODEL_NAME,'heart_acc': ml_service.HEART_ACCURACY,
    })
