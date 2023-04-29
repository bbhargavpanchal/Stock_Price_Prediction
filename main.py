from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    company = request.form['company']
    print(company)
    start = dt.datetime(2013, 1, 1)
    end = dt.datetime.now()
    data = yf.download(company, start=start, end=end)
    # print(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    test_start = dt.datetime(2022, 5, 1)
    test_end = dt.datetime.now()

    test_data = yf.download(company, start=test_start, end=test_end)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.plot(test_data.index, actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(test_data.index, predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time Durations')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    # plt.show()

    plot_filename = f'static/{company}_plot.png'
    plt.savefig(plot_filename)
    predicted_df = pd.DataFrame(predicted_prices, columns=['Predicted Price'])

    actual_df = pd.DataFrame(actual_prices, columns=['Actual Price'])
    merged_df = pd.concat([actual_df, predicted_df], axis=1)
    merged_df.index = test_data.index

    # Save the merged dataframe to a csv file
    csv_filename = f'static/{company}_predictions.csv'
    merged_df.to_csv(csv_filename)
    return render_template('result.html', company=company, prediction=predicted_prices[-1], plot_url=plot_filename)


if __name__ == '__main__':
    app.run(debug=True, port=5006)
