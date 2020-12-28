import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from numpy import inf
import math

app = Flask(__name__,static_url_path = "/tmp", static_folder = "tmp")
rf_model = pickle.load(open('fprc_r_rf.pkl', 'rb'))
ab_model = pickle.load(open('fprc_r_ab.pkl', 'rb'))
ann_model = pickle.load(open('fprc_r_ann.pkl','rb'))
cb_model = pickle.load(open('fprc_r_cb.pkl','rb'))
dt_model = pickle.load(open('fprc_r_dt.pkl','rb'))
knn_model = pickle.load(open('fprc_r_knn.pkl','rb'))
lasso_model = pickle.load(open('fprc_r_lasso.pkl','rb'))
lr_model = pickle.load(open('fprc_r_lr.pkl','rb'))
ridge_model = pickle.load(open('fprc_r_rr.pkl','rb'))
svr_model = pickle.load(open('fprc_r_svr.pkl','rb'))
xg_model = pickle.load(open('fprc_r_xb.pkl','rb'))


scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    int_features = [float(x) for x in request.form.values()]
    print(int_features)
    
    #print(int_features)
    b=int_features[0]
    d=int_features[1]
    fc=int_features[2]
    a_d=int_features[3]
    p=int_features[4]
    ef=int_features[5]
    
    #print(e1_do,e2_do,fu_fy,type_c)
    #print(int_features[5])
    final_features=[]
    #['a/d', 'pf', 'fc', 'd', 'bw', 'Ef', 'Vexp']
    
    
    #final_features=final_features+[np.log(e2_do)]
    #final_features=final_features+[np.log(fu_fy)]
    
    if a_d!=0:
        final_features=final_features+[np.log(a_d)]
    else:
        final_features=final_features+[(a_d)]
        
    if p!=0:
        final_features=final_features+[np.log(p)]
    else:
        final_features=final_features+[(p)]
        
    if fc!=0:
        final_features=final_features+[np.log(fc)]
    else:
        final_features=final_features+[(fc)]
        
    if d!=0:
        final_features=final_features+[np.log(d)]
    else:
        final_features=final_features+[(d)]
        
        
    if b!=0:
        final_features=final_features+[np.log(b)]
    else:
        final_features=final_features+[b]
        
    if ef!=0:
        final_features=final_features+[np.log(ef)]
    else:
        final_features=final_features+[ef]
    
    
  
    
    
    
    final_features = [np.array(final_features)]
    print(final_features)
    

    #final_features=np.log(final_features)

    #print(final_features)
    final_features=scaler.transform(final_features)
    
    
    print(final_features)
    
    
    rf_prediction = rf_model.predict(final_features)
    ab_prediction = ab_model.predict(final_features)
    ann_prediction= ann_model.predict(final_features)
    cb_prediction = cb_model.predict(final_features)
    dt_prediction=dt_model.predict(final_features)
    knn_prediction= knn_model.predict(final_features)
    lasso_prediction = lasso_model.predict(final_features)
    lr_prediction = lr_model.predict(final_features)
    ridge_prediction = ridge_model.predict(final_features)
    svr_prediction = svr_model.predict(final_features)
    xg_prediction = xg_model.predict(final_features)
    
    
    
    

    rf_prediction_o = math.exp(rf_prediction[0])
    ab_prediction_o = math.exp(ab_prediction[0])
    ann_prediction_o = math.exp(ann_prediction[0])
    cb_prediction_o = math.exp(cb_prediction[0])
    dt_prediction_o = math.exp(dt_prediction[0])
    knn_prediction_o = math.exp(knn_prediction[0])
    lasso_prediction_o=math.exp(lasso_prediction[0])
    lr_prediction_o = math.exp(lr_prediction[0])
    ridge_prediction_o = math.exp(ridge_prediction[0])
    svr_prediction_o = math.exp(svr_prediction[0])
    xg_prediction_o = math.exp(xg_prediction[0])
    
    rf_prediction_o=format(rf_prediction_o,'.3f')
    ab_prediction_o=format(ab_prediction_o,'.3f')
    ann_prediction_o=format(ann_prediction_o, '.3f')
    cb_prediction_o=format(cb_prediction_o, '.3f')
    dt_prediction_o=format(dt_prediction_o, '.3f')
    knn_prediction_o=format(knn_prediction_o, '.3f')
    lasso_prediction_o=format(lasso_prediction_o, '.3f')
    lr_prediction_o=format(lr_prediction_o, '.3f')
    ridge_prediction_o=format(ridge_prediction_o, '.3f')
    svr_prediction_o=format(svr_prediction_o, '.3f')
    xg_prediction_o=format(xg_prediction_o, '.3f')
    
    print(rf_prediction_o,ab_prediction_o,ann_prediction_o,cb_prediction_o,dt_prediction_o,knn_prediction_o,lasso_prediction_o,lr_prediction_o,ridge_prediction_o,svr_prediction_o,xg_prediction_o)
    

    
    return render_template('index.html', rf='{}'.format(rf_prediction_o), ab='{}'.format(ab_prediction_o),ann='{}'.format(ann_prediction_o),cb='{}'.format(cb_prediction_o), dt='{}'.format(dt_prediction_o), knn='{}'.format(knn_prediction_o), lasso='{}'.format(lasso_prediction_o), lr='{}'.format(lr_prediction_o), rr='{}'.format(ridge_prediction_o),svr='{}'.format(svr_prediction_o), xg='{}'.format(xg_prediction_o)) 


if __name__ == "__main__":
    app.run(debug=True)