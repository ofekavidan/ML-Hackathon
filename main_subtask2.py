import pandas as pd
import numpy as np
from argparse import ArgumentParser
import logging
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go



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
    df = X.copy()
    dropped_indexes = df[df.isna().any(axis=1)].index
    y = y.drop(dropped_indexes)
    df = df.dropna()

    non_first_station_indexes = df[df['station_index'] != 1].index
    if isinstance(y, pd.Series):
        y = y.drop(non_first_station_indexes)
    df = df[df['station_index'] == 1]
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
    # Create dummy variables for cluster and fill NaNs with 0
    df = pd.get_dummies(df, columns=['cluster'], dummy_na=False).fillna(0)

    # Create dummy variables for hours
    hours_dummies = pd.get_dummies(df['arrival_time'].dt.hour, prefix='hour')

    # Concatenate dummy variables with X
    df = pd.concat([df, hours_dummies], axis=1)
    if isinstance(y, pd.Series):
        df = df.set_index(y.index)

    # Drop unnecessary columns
    df.drop(columns=['trip_id', 'part', 'trip_id_unique_station', 'line_id', 'direction',
                     'alternative', 'station_index', 'station_id', 'station_name', 'door_closing_time',
                     'arrival_is_estimated', 'latitude', 'longitude', 'passengers_up', 'passengers_continue',
                     'mekadem_nipuach_luz', 'passengers_continue_menupach', 'trip_id_unique', 'arrival_time'], inplace=True)

    df = df.set_index(y.index)
    # Add the target variable back to the dataframe
    df['trip_duration'] = y

    # preprocess y:
    if isinstance(y, pd.Series):
        df = df.set_index(y.index)
        indexes_less_zero = y[y <= 0].index
        df = df.drop(indexes_less_zero)
        y = y.drop(indexes_less_zero)

    df = df.drop(columns=['trip_duration'])

    # Save the preprocessed data to a CSV file
    df.to_csv("outputfile.csv", index=False)
    return df, y


def preprocess_test(X: pd.DataFrame, y: pd.Series):
    df = X.copy()
    non_first_station_indexes = df[df['station_index'] != 1].index
    if isinstance(y, pd.Series):
        y = y.drop(non_first_station_indexes)
    df = df[df['station_index'] == 1]

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')

    # Create dummy variables for cluster and fill NaNs with 0
    df = pd.get_dummies(df, columns=['cluster'], dummy_na=False).fillna(0)

    # Create dummy variables for hours
    hours_dummies = pd.get_dummies(df['arrival_time'].dt.hour, prefix='hour')

    # Concatenate dummy variables with X
    df = pd.concat([df, hours_dummies], axis=1)

    # Drop unnecessary columns
    df.drop(columns=['trip_id', 'part', 'trip_id_unique_station', 'line_id', 'direction',
                     'alternative', 'station_index', 'station_id', 'station_name', 'door_closing_time',
                     'arrival_is_estimated', 'latitude', 'longitude', 'passengers_up', 'passengers_continue',
                     'mekadem_nipuach_luz', 'passengers_continue_menupach', 'trip_id_unique', 'arrival_time'], inplace=True)

    if isinstance(y, pd.Series):
        df = df.set_index(y.index)
    # Add the target variable back to the dataframe
    if isinstance(y, pd.Series):
        df['trip_duration'] = y

    # preprocess y:
    if isinstance(y, pd.Series):
        df = df.set_index(y.index)
        indexes_less_zero = y[y <= 0].index
        df = df.drop(indexes_less_zero)
        y = y.drop(indexes_less_zero)

    df = df.drop(columns=['trip_duration'])

    # Save the preprocessed data to a CSV file
    df.to_csv("outputfile2.csv", index=False)
    return df, y

def preprocess_out_test(X: pd.DataFrame, y: pd.Series):
    df = X.copy()
    df.index = df['trip_id_unique']
    non_first_station_indexes = df[df['station_index'] != 1].index
    if isinstance(y, pd.Series):
        y = y.drop(non_first_station_indexes)
    df = df[df['station_index'] == 1]

    df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')

    # Create dummy variables for cluster and fill NaNs with 0
    df = pd.get_dummies(df, columns=['cluster'], dummy_na=False).fillna(0)

    # Create dummy variables for hours
    hours_dummies = pd.get_dummies(df['arrival_time'].dt.hour, prefix='hour')

    # Concatenate dummy variables with X
    df = pd.concat([df, hours_dummies], axis=1)

    # Drop unnecessary columns
    df.drop(columns=['trip_id', 'part', 'trip_id_unique_station', 'line_id', 'direction',
                     'alternative', 'station_index', 'station_id', 'station_name', 'door_closing_time',
                     'arrival_is_estimated', 'latitude', 'longitude', 'passengers_up', 'passengers_continue',
                     'mekadem_nipuach_luz', 'passengers_continue_menupach', 'trip_id_unique', 'arrival_time'], inplace=True)

    if isinstance(y, pd.Series):
        df = df.set_index(y.index)
    # Add the target variable back to the dataframe
    if isinstance(y, pd.Series):
        df['trip_duration'] = y

    # preprocess y:
    if isinstance(y, pd.Series):
        df = df.set_index(y.index)
        indexes_less_zero = y[y <= 0].index
        df = df.drop(indexes_less_zero)
        y = y.drop(indexes_less_zero)


    # Save the preprocessed data to a CSV file
    df.to_csv("outputfile2.csv", index=False)
    return df, y


def calculate_trip_duration(df):
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])

    # Find the first and last arrival times for each trip_id_unique
    first_arrival_times = df[df['station_index'] == 1].set_index('trip_id_unique')['arrival_time']
    last_arrival_times = df.loc[df.groupby('trip_id_unique')['station_index'].idxmax()].set_index('trip_id_unique')['arrival_time']

    # Calculate the trip duration for each trip_id_unique in total seconds
    trip_durations_seconds = (last_arrival_times - first_arrival_times).dt.total_seconds()

    # Convert trip duration to minutes
    trip_durations_minutes = trip_durations_seconds / 60

    # Create a dataframe for the trip durations
    trip_durations_df = trip_durations_minutes.reset_index()
    trip_durations_df.columns = ['trip_id_unique', 'trip_duration']

    # Merge the trip durations with the original dataframe
    df = pd.merge(df, trip_durations_df, on='trip_id_unique')

    return df

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

    # 1. load the training set (args.training_set)
    df = pd.read_csv("data/HU.BER/train_bus_schedule.csv", encoding='ISO-8859-8')

    df_test = pd.read_csv("data/HU.BER/X_trip_duration.csv", encoding='ISO-8859-8')

    # Calculate trip durations
    df = calculate_trip_duration(df)

    df_test = calculate_trip_duration(df_test)

    # Split features and target
    X, y = df.drop("trip_duration", axis=1), df.trip_duration

    # Split train and test
    np.random.seed(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)

    # 2. preprocess the training set
    logging.info("Preprocessing train...")
    X_base, y_base = preprocess_train(X_train, y_train)
    X_test_for_base, y_test_for_base = preprocess_test(X_test, y_test)

    # -------------create a random model ----------------
    model = RandomModel()
    model.fit(X_base, y_base)
    y_pred_test_for_base = model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE[model] = float(mse)

    # Fit random forest regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_base, y_base)
    y_pred_test_for_base = model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE[model] = float(mse)

    # Fit Gradient Boosting regression model
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
    model.fit(X_base, y_base)
    y_pred_test_for_base = model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE[model] = float(mse)

    # Fit knn model
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_base, y_base)
    y_pred_test_for_base = knn_model.predict(X_test_for_base)
    mse = mean_squared_error(y_test_for_base, y_pred_test_for_base)
    model_MSE[model] = float(mse)

    logging.info("training...")
    # Extract model names and MSE values from the dictionary
    models = list(model_MSE.keys())  # list of names of models
    mse_values = list(model_MSE.values())  # list of the MSE's of each model
    min_mse_index = mse_values.index(min(mse_values))
    # Get the corresponding model name
    best_model = models[min_mse_index]  # the model

    # 4. load the test set (args.test_set)
    test_df = pd.read_csv(args.test_set, encoding='ISO-8859-8')

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test, _ = preprocess_out_test(test_df, None)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    y_pred_test = best_model.predict(X_test)
    # Combine trip_id_unique_station with predictions into a DataFrame

    output_df = pd.DataFrame({
        'trip_id_unique': X_test.index,
        'trip_duration_in_minutes': y_pred_test
    })
    output_path = args.out  # Path to the output file specified in command-line arguments
    output_df.to_csv(output_path, index=False)
    # Define a sunset color palette
    colors = ['#da8ea3', '#e3b6d0', '#8583b7', '#495692', '#2a5480', '#DD541C', '#C65F58']
    models_names = ["Random", "Random Forest", "Gradient_Boosting"]
    # Create a bar chart with customized colors and text labels
    fig = go.Figure([go.Bar(
        x=models_names,
        y=mse_values,
        marker_color=colors[:len(models)],
        text=[f'{mse:.2f}' for mse in mse_values],  # Format the text labels
        textposition='auto'
    )])

    # Create a bar chart with customized colors

    # Customize the layout
    fig.update_layout(
        title='Model vs MSE',
        xaxis_title='Model',
        yaxis_title='MSE',
        yaxis=dict(range=[0, max(mse_values) + 5]),  # Adjust the y-axis limit for better visualization
        template='plotly_white',
        width=500
    )
    fig.show()