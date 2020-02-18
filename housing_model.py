import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('toronto_housing.csv')
df['sqft'].isnull().value_counts()  # about 5000 with no sqft data (aka useless)
df = df.dropna()
# df.rename(columns={'final_price': 'price'})

df['city_district'].value_counts(ascending=True)[:20]
# df['city_district'].value_counts()
# df['city_district'].nunique()

df['type'].value_counts()
df.groupby('type').mean()

df = df[(df['type'] != 'Plex') & (df['type'] != 'Co-Ownership Apt') & (df['type'] != 'Co-Op Apt') & (df['type'] != 'Store W/Apt/Offc') & (df['type'] != 'Link')]
dummies = pd.get_dummies(df['type'], drop_first=True)
df = pd.concat([df.drop('type', axis=1), dummies], axis=1)
# df = df.drop('mean_district_income', axis=1)
dummies = pd.get_dummies(df['city_district'], drop_first=True)
df = pd.concat([df.drop('city_district', axis=1), dummies], axis=1)
# plt.figure(figsize=(12, 10))
# sns.scatterplot(df['long'], df['lat'], hue=df['final_price'])
# plt.show()

plt.figure(figsize=(15, 12))
sns.scatterplot('long', 'lat', hue='final_price', palette='coolwarm', data=df)
plt.show()

df = df.drop('long', axis=1)
df = df.drop('lat', axis=1)

# dropping over six million is redundant because dropping by type has already taken out the outliers
# df.sort_values('final_price', ascending=False).head(50)
# under_six_million = df[df['final_price'] < 6000000]
# plt.figure(figsize=(12, 10))
# sns.scatterplot('long', 'lat', hue='final_price', data=under_six_million)
# plt.show()


X = df.drop('final_price', axis=1).values
y = df['final_price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import joblib
joblib.dump(scaler, 'house_scaler.pkl')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()

model.add(Dense(151, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(151, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(151, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(151, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x=X_train,
          y=y_train,
          validation_data=(X_test, y_test),
          batch_size=256,
          epochs=400,
          callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)

# without districts -> with -> with district and no long/lat
df['final_price'].mean()  # $761413
np.sqrt(mean_squared_error(predictions, y_test))  # $246650 -> $215141 -> $186960
mean_absolute_error(predictions, y_test)  # $158694 -> $117412 -> $101199
explained_variance_score(y_test, predictions)  # 0.748 -> 0.828 -> 0.869

model.save('toronto_housing_model_final.h5')

from tensorflow.keras.models import load_model
model = load_model('toronto_housing_model.h5')
# unscaled_sample = scaler.inverse_transform([X_train[0]])


# model.save('toronto_housing_model (districts).h5')
# X_train[0]
# model.predict(np.array([X_test[100]]))
# y_test[100]
#
# model.predict(np.array([X_test[362]]))
# y_test[362]
# len(X_test)
#
# '''
# sqft
# housing type
# bathrooms
# parking
# bedrooms normal
# bedrooms special
# city district
# '''
# scaler.inverse_transform([X_test[100]])