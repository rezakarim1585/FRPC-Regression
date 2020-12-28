import requests


#['a/d', 'pf', 'fc', 'd', 'bw', 'Ef', 'Vexp']

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'a_d':1, 'pf':1, 'fc':1, 'd' : 0, 'b' : 1, 'ef' : 1})

print(r.json())