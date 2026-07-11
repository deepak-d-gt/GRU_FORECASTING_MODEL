import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.regularizers import l2

from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

data = pd.read_csv('C:\\Users\\Deepak Durairaj\\Downloads\\GRU_FORECASTING_MODEL\\data\\Training Dataset1.csv')

data['KEY_DATE'] = pd.to_datetime(
    data['KEY_DATE'],
    format='%d.%m.%Y'
)

data = data.drop_duplicates()

data['GROWTH_PERCENT'] = (
    data['GROWTH_PERCENT']
    .replace('-', np.nan)
)

data['GROWTH_PERCENT'] = (
    data['GROWTH_PERCENT']
    .ffill()
)

data['GROWTH_PERCENT'] = pd.to_numeric(
    data['GROWTH_PERCENT']
)

features = data[['AMOUNT', 'SHOCK', 'IMPACT']]
target = data[['GROWTH_PERCENT']]

training_size = int(len(data) * 0.80)

X_train_raw = features[:training_size]
X_test_raw = features[training_size:]

y_train_raw = target[:training_size]
y_test_raw = target[training_size:]



X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train_scaled = X_scaler.fit_transform(X_train_raw)
X_test_scaled = X_scaler.transform(X_test_raw)

y_train_scaled = y_scaler.fit_transform(y_train_raw)
y_test_scaled = y_scaler.transform(y_test_raw)

timesteps = 160

X_train = []
y_train = []

for i in range(timesteps, len(X_train_scaled)):
    X_train.append(
        X_train_scaled[i-timesteps:i]
    )
    y_train.append(
        y_train_scaled[i]
    )

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = []
y_test = []

for i in range(timesteps, len(X_test_scaled)):
    X_test.append(
        X_test_scaled[i-timesteps:i]
    )
    y_test.append(
        y_test_scaled[i]
    )

X_test = np.array(X_test)
y_test = np.array(y_test)



regressor = Sequential()

regressor.add(
    GRU(
        units=64,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )
)

regressor.add(Dropout(0.14673260028436005))

regressor.add(
    GRU(
        units=32,
    )
)

regressor.add(Dropout(0.14673260028436005))

regressor.add(Dense(1))

optimizer = Adam(
    learning_rate=0.0015477558716539422
)

regressor.compile(
    optimizer=optimizer,
    loss='mean_squared_error'
)



early_stop = EarlyStopping(
    monitor='loss',
    patience=20,
    restore_best_weights=True
)

history = regressor.fit(
    X_train,
    y_train,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)



y_pred = regressor.predict(X_test)

y_pred = y_scaler.inverse_transform(
    y_pred
)

y_test_actual = y_scaler.inverse_transform(
    y_test.reshape(-1, 1)
)



rmse = np.sqrt(
    mean_squared_error(
        y_test_actual,
        y_pred
    )
)

mae = mean_absolute_error(
    y_test_actual,
    y_pred
)

r2 = r2_score(
    y_test_actual,
    y_pred
)

print("RMSE =", rmse)
print("MAE  =", mae)
print("R²   =", r2)

# =========================
# SAVE RESULTS
# =========================

results = pd.DataFrame({
    "Actual_Growth": y_test_actual.flatten(),
    "Predicted_Growth": y_pred.flatten()
})

results.to_csv(
    "GRU_Predictions.csv",
    index=False
)

print("Predictions saved successfully.")