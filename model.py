import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
import os

plt.style.use('bmh')

market = pd.read_csv("adoni_market.xls")

kapas = market.loc[market['Variety'] == 'Kapas (Adoni)']

kapas = kapas[['Arrival_Date','Min Price','Max Price','Modal Price']]

#Dataset after 2500 rows is continuous i.e very less dates are missing
#DATA FROM YEAR 2011
kapas = kapas.loc[2500: , :]
kapas.reset_index(drop=True)
kapas.set_index('Arrival_Date')
# kapas['Arrival_Date'] = pd.to_datetime(kapas['Arrival_Date'])

plt.figure(figsize=(20,10))
plt.plot(kapas["Modal Price"])
# plt.plot(kapas["Min Price"])
# plt.plot(kapas["Max Price"])
plt.title('Cotton Prices')
plt.ylabel('Price (INR)')
plt.xlabel('Days')
# plt.legend(['Modal Price', 'Min Price', 'Max Price'], loc='upper right')
plt.show()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

kapas_modal = kapas.iloc[:, 3:].values
#saving data to disk for future use
path = 'C:\\Users\\kaush\\Desktop\\Cotton Price prediction'
kapas.iloc[:, 3:].to_csv(os.path.join(path,r'kapas_modal.csv'),index=False)
training_set, test_set = train_test_split(kapas_modal, train_size=0.8, test_size=0.2, shuffle=False)

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1892):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 25))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#saving model to the disk
pickle.dump(regressor, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

inputs = kapas_modal[len(kapas_modal) - len(test_set) - 60 : ]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 534):
    X_test.append(inputs[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_cotton_price = regressor.predict(X_test)
predicted_cotton_price = sc.inverse_transform(predicted_cotton_price)

rmse = sqrt(mean_squared_error(test_set, predicted_cotton_price))
rmse

#Plotting the data
plt.figure(figsize=(20,10))
plt.plot(test_set, color= 'green', label = 'Real modal price')
plt.plot(predicted_cotton_price, color = 'red', label = 'Predicted Modal Price')
plt.title('Predicted Modal Price')
plt.xlabel('Days')
plt.ylabel('Cotton modal price')
plt.legend(fontsize=18)
plt.show()

def future_prediction(days):
    kapas_modal_future = kapas_modal.copy()
    for i in range(days):
        last_60_days = kapas_modal_future[-60:]
        last_60_days_scaled = sc.transform(last_60_days)
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_price = regressor.predict(X_test)
        pred_price = sc.inverse_transform(pred_price)
        kapas_modal_future = np.append(kapas_modal_future, pred_price, axis=0)
        future_prices = kapas_modal_future[-days:]
#         print('prediction for day ', i , ": ", pred_price)
    return pred_price

future_prediction(4)

model = pickle.load(open('model.pkl', 'rb'))
model.predict()











