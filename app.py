import numpy as np
import pandas as pd
from flask import Flask ,request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import backend as K


app = Flask(__name__)

def load_model():
    global model
    K.clear_session()
    model = pickle.load(open('model2.pkl', 'rb'))
    global graph
    graph = tf.get_default_graph()
    


def preprocessing_data(days):
    sc = MinMaxScaler(feature_range = (0, 1))
    kapas = pd.read_csv("kapas_modal.csv")
    kapas_modal = kapas.values
    kapas_modal_future = kapas_modal.copy()
    load_model()
    for i in range(days):
        last_60_days = kapas_modal_future[-60:]
        last_60_days_scaled = sc.fit_transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        #load_model()
        with graph.as_default():
            pred_price = model.predict(X_test)
            pred_price = sc.inverse_transform(pred_price)
        kapas_modal_future = np.append(kapas_modal_future, pred_price, axis=0)
        #future_prices = kapas_modal_future[-days:]        
    return pred_price

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    for x in request.form.values():
        days = int(x)
    
    prediction = preprocessing_data(days)
    return render_template('index.html', prediction_text='Predicted cotton price is for day {} is {}'.format(days, round(prediction[0][0]), 3))


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    