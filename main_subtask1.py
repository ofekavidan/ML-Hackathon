from argparse import ArgumentParser
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

"""
usage:
    python code/main_subtask1.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main_subtask1.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


# implement here your load,preprocess,train,predict,save functions (or any other design you choose)

def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """
    # Keep only the required columns
    X.index = X['trip_id_unique_station']
    if isinstance(y, pd.Series):
        y.index = X.index
    X = X[['arrival_time', 'station_index', 'cluster']]

    # Fill null values in 'arrival_time' with the average time
    if X['arrival_time'].isnull().any():
        average_time = X['arrival_time'].mean()
        X['arrival_time'].fillna(average_time, inplace=True)

    # Convert 'arrival_time' to datetime
    X['arrival_time'] = pd.to_datetime(X['arrival_time'], format='%H:%M:%S', errors='coerce')

    # Create 'is_rush_hour' column
    X['is_rush_hour'] = X['arrival_time'].apply(
        lambda x: 1 if ((7 <= x.hour < 9) or (16 <= x.hour < 18)) else 0)

    # Create dummy variables for hours
    hours_dummies = pd.get_dummies(X['arrival_time'].dt.hour, prefix='hour')

    # Concatenate dummy variables with X
    X = pd.concat([X, hours_dummies], axis=1)

    # Drop 'arrival_time' and 'is_first_station' columns
    X.drop(columns=['arrival_time', 'station_index'], inplace=True)

    # 'cluster' is a categorical column -> to dummies vars
    X = pd.get_dummies(X, columns=['cluster'])

    # preprocess y:
    if isinstance(y, pd.Series):
        y = y.dropna()
        y = y[y > 0]
        X = X.loc[y.index]  # remove corresponding rows from X

    return X, y


def add_trip_id_column(preprocessed_df, column_to_add):
    # 2. Align indices
    preprocessed_df = preprocessed_df.loc[preprocessed_df.index]

    # 3. Insert the column back to the preprocessed DataFrame as the first column
    preprocessed_df.insert(0, column_to_add.name, column_to_add)

    return preprocessed_df


class RandomModel:
    def fit(self, X, y):
        # Store the number of samples for generating random predictions
        self.num_samples = len(y)
        # Store the range of target values for generating random predictions
        self.y_min = y.min()
        self.y_max = y.max()

    def predict(self, X):
        # Generate random predictions within the range of the target values
        return np.random.uniform(self.y_min, self.y_max, size=len(X))


def create_and_fit_random_model(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base):
    model = RandomModel()
    model.fit(X_base, y_base)
    y_pred_test_for_base = model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE['random_model'] = float(mse)
    return model

def fit_polynomial_regression(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base):
    # Fit polynomial regression models of different degrees
    # for degree in [1, 2, 3, 4, 5]:
    for degree in [1]:
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_base, y_base)
        y_pred_test_for_base = model.predict(X_test_for_base)
        mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
        model_MSE['polynomial_regression_deg_' + str(degree)] = float(mse)
    return model


def fit_random_forest(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_base, y_base)
    y_pred_test_for_base = model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE['random_forest'] = float(mse)
    return model


def fit_gradient_boosting(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base):
    # Fit Gradient Boosting regression model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    model.fit(X_base, y_base)
    y_pred_test_for_base = model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE['gradient_boosting'] = float(mse)
    return model

def fit_KNN(model_MSE,X_base,y_base,X_test_for_base,y_test_for_base):
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_base, y_base)
    y_pred_test_for_base = knn_model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE['KNN'] = float(mse)
    return knn_model

def fit_SVR(model_MSE,X_base,y_base,X_test_for_base,y_test_for_base):
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.2)
    svr_model.fit(X_base, y_base)
    y_pred_test_for_base = svr_model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE['SVR'] = float(mse)
    return svr_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    model_MSE = dict()
    np.random.seed(0)

    # 1. load the training set (args.training_set)
    df = pd.read_csv(args.training_set, encoding='ISO-8859-8')
    X, y = df.drop("passengers_up", axis=1), df.passengers_up

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    # split the data - 5%  training, 95% testing
    X_base, X_test_for_base, y_base, y_test_for_base = train_test_split(X, y, train_size=0.8, random_state=0)
    # X_test_for_base, X_temp2, y_test_for_base, y_temp2 = train_test_split(X, y, train_size=0.2, random_state=42)

    # Store the original column before pre-processing
    # preprocess train and test
    X_base, y_base = preprocess_train(X_base, y_base)
    X_test_for_base, y_test_for_base = preprocess_train(X_test_for_base, y_test_for_base)

    # 3. train a model

    # -------------create a random model ----------------
    models=[]
    rand_model =create_and_fit_random_model(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base)
    models.append(rand_model)
    poly_model=fit_polynomial_regression(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base)
    models.append(poly_model)
    # Fit random forest regressor model
    random_forest_model=fit_random_forest(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base)
    models.append(random_forest_model)
    # Fit Gradient Boosting regression model
    gb_model=fit_gradient_boosting(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base)
    models.append(gb_model)
    # Fit KNN model
    knn_model=fit_KNN(model_MSE, X_base, y_base, X_test_for_base, y_test_for_base)
    models.append(knn_model)

    # Extract model names and MSE values from the dictionary
    models_keys = list(model_MSE.keys())
    mse_values = list(model_MSE.values())

    # Define a sunset color palette
    colors = ['#da8ea3', '#e3b6d0', '#8583b7', '#495692', '#2a5480', '#DD541C', '#C65F58']

    # Assuming models_keys, mse_values, and colors are already defined
    fig = go.Figure([go.Bar(
        x=models_keys,
        y=mse_values,
        marker_color=colors[:len(models_keys)],
        text=mse_values,  # Add text for the values
        textposition='auto'  # Position the text automatically
    )])

    # Customize the layout
    fig.update_layout(
        title='Model vs MSE',
        xaxis_title='Model',
        yaxis_title='MSE',
        yaxis=dict(range=[0, max(mse_values) + 5]),  # Adjust the y-axis limit for better visualization
        template='plotly_white',
        width=800
    )

    fig.show()

    # Load the data
    df = pd.read_csv("data/HU.BER/train_bus_schedule.csv", encoding='ISO-8859-8')

    # Calculate the average number of passengers for each bus line
    line_avg_passengers = df.groupby('line_id')['passengers_up'].mean().reset_index()

    # Sort the bus lines by the average number of passengers
    line_avg_passengers = line_avg_passengers.sort_values(by='passengers_up', ascending=False)

    # Take the top 5 bus lines with the highest average number of passengers
    top_lines = line_avg_passengers.head(5)

    # Create the bar chart
    fig_top_lines = go.Figure()
    fig_top_lines.add_trace(go.Bar(
        x=top_lines['line_id'].astype(str),
        y=top_lines['passengers_up'],
        marker_color='#FFA07A'
    ))

    fig_top_lines.update_layout(
        title='Top 5 Bus Lines by Average Number of Passengers Boarding',
        xaxis_title='Bus Line',
        yaxis_title='Average Number of Passengers',
        template='plotly_white',
        width=800
    )

    fig_top_lines.show()

    # Load the data
    df = pd.read_csv("data/HU.BER/train_bus_schedule.csv", encoding='ISO-8859-8')

    # Calculate the average number of passengers for each station
    station_avg_passengers = df.groupby('station_id')['passengers_up'].mean().reset_index()

    # Sort the stations by the average number of passengers
    station_avg_passengers = station_avg_passengers.sort_values(by='passengers_up', ascending=False)

    # Take the top 5 stations with the highest average number of passengers
    top_stations = station_avg_passengers.head(5)

    # Create the bar chart
    fig_top_stations = go.Figure()
    fig_top_stations.add_trace(go.Bar(
        x=top_stations['station_id'].astype(str),
        y=top_stations['passengers_up'],
        marker_color='#90EE90'
    ))

    fig_top_stations.update_layout(
        title='Top 5 Stations by Average Number of Passengers Boarding',
        xaxis_title='Station ID',
        yaxis_title='Average Number of Passengers',
        template='plotly_white',
        width=800
    )

    fig_top_stations.show()

    logging.info("training...")

    min_mse_index = mse_values.index(min(mse_values))
    # Get the corresponding model name
    best_model = models[min_mse_index]


    # 4. load the test set (args.test_set)
    test_df = pd.read_csv(args.test_set, encoding='ISO-8859-8')

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    # X_test = test_df.drop(columns=['passengers_up'])
    X_test, _ = preprocess_train(test_df, None)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    y_pred_test = best_model.predict(X_test)
    y_pred_test = np.where(y_pred_test - np.floor(y_pred_test) >= 0.5, np.ceil(y_pred_test),
                           np.floor(y_pred_test)).astype(int)
    # X_test_for_base = add_trip_id_column(X_test_for_base, trip_id_unique_station)
    # Combine trip_id_unique_station with predictions into a DataFrame
    output_df = pd.DataFrame({
        'trip_id_unique_station': X_test.index,
        'passengers_up': y_pred_test.flatten()
    })
    output_path = args.out  # Path to the output file specified in command-line arguments
    output_df.to_csv(output_path, index=False)


    # Additional plot: Average number of passengers during rush hours vs non-rush hours
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
    df['is_rush_hour'] = df['arrival_time'].apply(lambda x: 1 if ((7 <= x.hour < 9) or (16 <= x.hour < 18)) else 0)

    rush_hour_avg = df[df['is_rush_hour'] == 1]['passengers_up'].mean()
    non_rush_hour_avg = df[df['is_rush_hour'] == 0]['passengers_up'].mean()

    fig_rush_hour = go.Figure()
    fig_rush_hour.add_trace(go.Bar(
        x=['Rush Hour', 'Non-Rush Hour'],
        y=[rush_hour_avg, non_rush_hour_avg],
        marker_color=['#DD541C', '#2a5480']
    ))

    fig_rush_hour.update_layout(
        title='Average Number of Passengers Boarding During Rush Hours vs Non-Rush Hours',
        xaxis_title='Time of Day',
        yaxis_title='Average Number of Passengers',
        template='plotly_white',
        width=800
    )
    fig_rush_hour.show()
