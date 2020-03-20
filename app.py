import numpy as np
import pandas as pd
from flask import Flask ,request, jsonify, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras import backend as K
from datetime import date


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
    global prices
    prices = []
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
        prices.append(int(pred_price[0][0])) 
    #prices = pd.DataFrame(prices)

    #prices.to_csv("predicted_prices.csv")       
    return pred_price

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data')
def data():
    counts = list(range(1, days+1,1))
    return jsonify({'prices': prices,
                    'days' : counts})

@app.route('/predict',methods=['POST'])
def predict():
    global days
    global  tod_date
    
    for x in request.form.values():
        year=int(x[:4])
        month=int(x[5:7])
        day=int(x[8:10])
        fut_date=date(year,month,day)
        tod_date=date.today()
        delta = fut_date - tod_date
        days=delta.days
    if month==1:
        m="January"
    elif month==2:
        m="February"
    elif month==3:
        m="March"
    elif month==4:
        m="April"
    elif month==5:
        m="May"
    elif month==6:
        m="June"
    elif month==7:
        m="July"
    elif month==8:
        m="August"
    elif month==9:
        m="September"
    elif month==10:
        m="October"
    elif month==11:
        m="November"
    elif month==12:
        m="December"    
    prediction = preprocessing_data(days)
    return render_template('index.html', prediction_text='Predicted Cotton Price for {} {}, {} is Rs. {}0'.format(day,m,year, round(prediction[0][0]), 3))

@app.route('/modelacc')
def modelacc():
    return render_template('realvspred.html')

@app.route("/chart", methods=['POST'])
def chart():
    counts = []
    for i in range(1, days+1,1):
        counts.append("Day " + str(i) )
    legend = 'Future Price Trend'
    labels = counts
    values = prices
    return render_template('chart.html', values=values, labels=labels, legend=legend)


@app.route('/analysis')
def analysis():
    return render_template('analysis.html')


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
    
    
    
    