import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('toronto_housing.csv')

# drop rows with missing data
df['sqft'].isnull().value_counts()  # about 5000 with no sqft data
df = df.dropna()

# get some insight on the dataset
df['city_district'].value_counts(ascending=True)[:20]
df['city_district'].value_counts()
df['city_district'].nunique()
df['type'].value_counts()
df.groupby('type').mean()

# drop housing types that have very low value count
df = df[(df['type'] != 'Plex') & (df['type'] != 'Co-Ownership Apt') & (df['type'] != 'Co-Op Apt') & (df['type'] != 'Store W/Apt/Offc') & (df['type'] != 'Link')]

# one-hot encode housing type and city district
dummies = pd.get_dummies(df['type'], drop_first=True)
df = pd.concat([df.drop('type', axis=1), dummies], axis=1)
dummies = pd.get_dummies(df['city_district'], drop_first=True)
df = pd.concat([df.drop('city_district', axis=1), dummies], axis=1)

# create a scatter plot for data visualization
plt.figure(figsize=(12, 10))
sns.scatterplot(df['long'], df['lat'], hue=df['final_price'])
plt.show()

plt.figure(figsize=(15, 12))
sns.scatterplot('long', 'lat', hue='final_price', palette='coolwarm', data=df)
plt.show()

df = df.drop('long', axis=1)
df = df.drop('lat', axis=1)

# data pre-processing
X = df.drop('final_price', axis=1).values
y = df['final_price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save the scaler for later use
import joblib
joblib.dump(scaler, 'house_scaler.pkl')

# create the ANN
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

# stop training if val_loss doesn't decrease in two epochs
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

model.fit(x=X_train,
          y=y_train,
          validation_data=(X_test, y_test),
          batch_size=256,
          epochs=400,
          callbacks=[early_stop])

# plot losses
losses = pd.DataFrame(model.history.history)
losses.plot()
plt.show()

# evaluate model performance
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
predictions = model.predict(X_test)

df['final_price'].mean()  # $761413
np.sqrt(mean_squared_error(predictions, y_test))  # $186960
mean_absolute_error(predictions, y_test)  # $101199
explained_variance_score(y_test, predictions)  # 0.869

# save the model for later use
model.save('toronto_housing_model_final.h5')
