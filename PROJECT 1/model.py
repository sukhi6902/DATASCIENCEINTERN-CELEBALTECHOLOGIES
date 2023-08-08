# Importing the libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# Importing the pre-processed excel file
dataset = pd.read_excel('cleaned_city_day.xlsx')

# Splitting Training and Testing Data
# Seperate the feature columns and target column
features = dataset[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'NOx', 'NO']]
target = dataset['AQI']

# Split the data into training(80%) and testing (20%)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42)

# Model Creation - Taking RandomForest as it showed the maximum accuracy


# Initializing the model
forestgump = RandomForestRegressor()

# Training the model
forestgump.fit(X_train, y_train)

# Saving model to disk using pickle
pickle.dump(forestgump, open('model.pkl', 'wb'))


# FOR TESTING

# Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[12.3,4.5,34.6,56.7,7.6,12.3,12.5]]))
