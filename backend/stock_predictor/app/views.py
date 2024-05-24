from django.shortcuts import render
from django.http import JsonResponse
import yfinance as yf
from stock_predictor.predict import run_model

def get_historical_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1y")  # Fetch 1 year of data
    return hist

def index(request):
    return render(request, 'index.html')

def predict_view(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        historical_data = get_historical_data(symbol)
        result = run_model(historical_data)  # Run the model and get the result dictionary

        return render(request, 'index.html', {'result': result, 'symbol': symbol})
    return JsonResponse({'error': 'Invalid request method'}, status=400)







