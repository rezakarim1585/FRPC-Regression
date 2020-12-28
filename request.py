import requests


#['a/d', 'pf', 'fc', 'd', 'bw', 'Ef', 'Vexp']

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'a_d':3, 'pf':4, 'fc':2, 'd' : 1, 'b' : 0, 'ef' : 5})

print(r.json())
