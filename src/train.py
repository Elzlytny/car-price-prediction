# Machine Learning Project Regression algorithms.


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Reading the dataset
dataSet = pd.read_csv(r'car details v4.csv')

# Data Preprocessing
dataSet['Engine'] = dataSet['Engine'].str.replace(' cc', '', regex=False).astype(float)
dataSet['Max Power'] = (dataSet['Max Power'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float))
dataSet['Max Torque'] = (dataSet['Max Torque'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float))
owner_map = {
    'First': 1,
    'Second': 2,
    'Third': 3,
    'Fourth & Above': 4
}
dataSet['Owner'] = dataSet['Owner'].map(owner_map)
num_cols = [
    'Owner', 'Engine', 'Max Power', 'Max Torque',
    'Length', 'Width', 'Height',
    'Seating Capacity', 'Fuel Tank Capacity'
]

for col in num_cols:
    dataSet[col] = dataSet[col].fillna(dataSet[col].median())

cat_cols = ['Drivetrain']

for col in cat_cols:
    dataSet[col] = dataSet[col].fillna(dataSet[col].mode()[0])


# Splitting dataset into features and target variable
X = dataSet.drop(['Price', 'Model'], axis='columns')
X = pd.get_dummies(X, drop_first=True)
Y = dataSet['Price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 1 - Linear Regression algorithm
LRmodel = LinearRegression()
LRmodel.fit(X_train, Y_train)
train_predictions = LRmodel.predict(X_train)
train_mse = mean_squared_error(Y_train, train_predictions)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(Y_train, train_predictions)
print(f'Linear Regression Training :\nMSE: {train_mse}, \nRMSE: {train_rmse}, \nR2: {train_r2}')

test_predictions = LRmodel.predict(X_test)
test_mse = mean_squared_error(Y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(Y_test, test_predictions)
print(f'Linear Regression Test :\nMSE: {test_mse}, \nRMSE: {test_rmse}, \nR2: {test_r2}')


# 2 - K-Nearest Neighbors Regression algorithm
KNN = KNeighborsRegressor(n_neighbors=3)
KNN.fit(X_train_scaled, Y_train)
knn_train_predictions = KNN.predict(X_train_scaled)
train_mse = mean_squared_error(Y_train, knn_train_predictions)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(Y_train, knn_train_predictions)
print(f'KNN Training :\nMSE: {train_mse}, \nRMSE: {train_rmse}, \nR2: {train_r2}')

knn_test_predictions = KNN.predict(X_test_scaled)
test_mse = mean_squared_error(Y_test, knn_test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(Y_test, knn_test_predictions)
print(f'KNN Test :\nMSE: {test_mse}, \nRMSE: {test_rmse}, \nR2: {test_r2}')



# 3 - Random Forest Regression algorithm
RFmodel = RandomForestRegressor(n_estimators=100, random_state=42)
RFmodel.fit(X_train, Y_train)
train_predictions = RFmodel.predict(X_train)
train_mse = mean_squared_error(Y_train, train_predictions)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(Y_train, train_predictions)
print(f'Random Forest Training :\nMSE: {train_mse}, \nRMSE: {train_rmse}, \nR2: {train_r2}')

test_predictions = RFmodel.predict(X_test)
test_mse = mean_squared_error(Y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(Y_test, test_predictions)
print(f'Random Forest Test :\nMSE: {test_mse}, \nRMSE: {test_rmse}, \nR2: {test_r2}')


# 4 - Support Vector Regression algorithm
SVmodel = SVR()
SVmodel.fit(X_train_scaled, Y_train)
train_predictions = SVmodel.predict(X_train_scaled)
train_mse = mean_squared_error(Y_train, train_predictions)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(Y_train, train_predictions)
print(f'Support Vector Regression Training :\nMSE: {train_mse}, \nRMSE: {train_rmse}, \nR2: {train_r2}')

test_predictions = SVmodel.predict(X_test_scaled)
test_mse = mean_squared_error(Y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(Y_test, test_predictions)
print(f'Support Vector Regression Test :\nMSE: {test_mse}, \nRMSE: {test_rmse}, \nR2: {test_r2}')

# 5 - Decision Tree Regression algorithm
DTmodel = DecisionTreeRegressor()
DTmodel.fit(X_train, Y_train)
train_predictions = DTmodel.predict(X_train)
train_mse = mean_squared_error(Y_train, train_predictions)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(Y_train, train_predictions)
print(f'Decision Tree Training :\nMSE: {train_mse}, \nRMSE: {train_rmse}, \nR2: {train_r2}')

test_predictions = DTmodel.predict(X_test)
test_mse = mean_squared_error(Y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(Y_test, test_predictions)
print(f'Decision Tree Test :\nMSE: {test_mse}, \nRMSE: {test_rmse}, \nR2: {test_r2}')

# Saving the best model (Random Forest Regression) and the columns used in the model for making GUI
import joblib
joblib.dump(RFmodel, 'rf_model.pkl')
joblib.dump(X.columns, 'model_columns.pkl')

# Conclusion:
# After evaluating all the regression algorithms, we can compare their performance based on the MSE, RMSE, and R2 scores.
# Here are the summarized results:
    # 1 - Linear Regression:
        # Linear Regression Training :
            # MSE: 1253320423170.1877, RMSE: 1119517.9423172225, R2: 0.7749582510694774      
        # Linear Regression Test :  
            # MSE: 2486749273956.3423, RMSE: 1576943.0154435963, R2: 0.6440525580522786

    # 2 - KNN Regression:
        # KNN Training :
            # MSE: 1077319172093.9381, RMSE: 1037939.8692091648, R2: 0.8065604085257282    
        # KNN Test :
            # MSE: 3151423534285.93, RMSE: 1775224.9249844174, R2: 0.5489126478206352

    # 3 - Random Forest Regression:
        # Random Forest Training :  
            # MSE: 100531676303.25694, RMSE: 317067.30563597527, R2: 0.9819488904513527    
        # Random Forest Test :
            # MSE: 1210211436871.2615, RMSE: 1100096.1034706293, R2: 0.8267731814857002

    # 4 - Support Vector Regression:
        # Support Vector Regression Training :
            # MSE: 6328909502554.537, RMSE: 2515732.398836279, R2: -0.13639643697442283
        # Support Vector Regression Test :
            # MSE: 7802124760239.243, RMSE: 2793228.3759548273, R2: -0.11677778666676075

    # 5 - Decision Tree Regression:
        # Decision Tree Training :
            # MSE: 877663934.4262295, RMSE: 29625.39340542551, R2: 0.9998424097915225
        # Decision Tree Test :
            # MSE: 966334443558.3423, RMSE: 983023.114457815, R2: 0.8616811606811772

# From the results, we can see that the Random Forest Regression algorithm performed the best on the test set with an R2 score of approximately 0.81, indicating a good fit.
# The Decision Tree also performed well but showed signs of overfitting with a very high training R2 score.
# Linear Regression and KNN had moderate performance, while Support Vector Regression did not perform well on this dataset.