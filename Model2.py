import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
pd.options.mode.copy_on_write = True


def NBA_df_config():
    """ Function to create NBA data frame including desired features and win percentages """

    NBA_df = pd.read_csv("pergame_stats_total.csv")

    # Customize to include wanted features
    NBA_df = NBA_df[["Team", "FGA", "FG_Percent", "3PA", "3P_Percent", "FTA", "FT_Percent", "ORB", "DRB", "AST",
                    "STL", "BLK", "TOV", "PF", "Year"]]

    NBA_df = NBA_df[NBA_df["Team"] != "League Average"]

    # Find win percentages of each year
    win_percentage_df = pd.read_csv("advanced_stats_total.csv")
    win_percentage_df = win_percentage_df[["Team", "W", "L", "Year"]]

    winPctTeam = {}
    # don't want first row because they are just column headers
    for index, row in win_percentage_df.iterrows():

        if (row.iloc[0] != "League Average") and (row.iloc[0] != "Team"):

            team = row.iloc[0]
            wins = int(row.iloc[1])
            losses = int(row.iloc[2])
            year = int(row.iloc[3])
            win_pct = wins / (wins + losses)
            teamYr = (team, year)
            winPctTeam[teamYr] = win_pct


    NBA_df["WIN%"] = 0.0

    # Update NBA_df to include win percentages
    for index, row in NBA_df.iterrows():
        year = NBA_df.at[index, "Year"]
        if (row.iloc[0], year) in winPctTeam:
            NBA_df.loc[index, "WIN%"] = winPctTeam[(row["Team"], year)]

    # Check for missing win percentages
    NBA_df = NBA_df.dropna(subset=["WIN%"])

    # Remove year column so df is identical to wNBA 
    NBA_df = NBA_df.drop(columns=['Year'])

    return NBA_df

def wNBA_df_config():
    """ Function to create NBA data frame including desired features and win percentages """


    wNBA_df = pd.read_csv("WNBA Statistics.csv")

    # Customize data frame to include wanted features
    wNBA_df = wNBA_df[["Team", "FGA", "FG_Percent", "3PA", "3P_Percent", "FTA", "FT_Percent", "ORB", "DRB", "AST",
                 "STL", "BLK", "TOV", "PF", "WIN%"]]

    # Don't include any data not associated with a specific team
    wNBA_df = wNBA_df[wNBA_df["Team"] != "League Average"]

    return wNBA_df

def regression(df):
    """ This function runs a regression on the feature data. It finds coefficients and r2 values as well."""

    # Run Regression 
    X = df[["FGA", "FG_Percent", "3PA", "3P_Percent", "FTA", "FT_Percent", "ORB", "DRB", "AST",
                 "STL", "BLK", "TOV", "PF"]]
    y = df['WIN%']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features of X_train and X_test    

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)    
    X_test = scaler.transform(X_test)

    # track r2 for each feature
    r2Final = []

    # track coefficients
    coefficients = []
    
    for i, feature in enumerate(X.columns):

        model = LinearRegression()

        # run regression on single feature at a time, have to reshape vectors
        x_train_feature = np.array(X_train[:, i]).reshape(-1, 1)
        x_test_feature = np.array(X_test[:, i]).reshape(-1, 1)

        model.fit(x_train_feature, y_train)

        # compare prediction with actual values using r2_score method
        prediction = model.predict(x_test_feature)
        r2 = r2_score(y_test, prediction)

        coeff = model.coef_

        r2Final.append(r2)
        coefficients.append(coeff[0])
    
    overallModel = LinearRegression()
    overallModel.fit(X_train, y_train)
    overall_predictions = overallModel.predict(X_test)

    overall_r2 = r2_score(y_test, overall_predictions)
    overall_mse = mean_squared_error(y_test, overall_predictions)

    # make overall dataframe
    overallData = {'Metric': ['R2 Score', 'MSE'], '': [overall_r2, overall_mse]}
    overallDataFrame = pd.DataFrame(overallData)


    # make feature dataframe
    featureData = {'Feature': X.columns, 'R2': r2Final, 'Coef': coefficients}

    featureDataFrame = pd.DataFrame(featureData)
    return (featureDataFrame, overallDataFrame, overallModel)

def wNBA_2024():
    """ This function runs a prediction of win percentages for the 2024 WNBA season based on our model. Then 
    the predicted win percentage is compared with the actual win percentage to calculate percent error as well. """
    
    scaler = StandardScaler()
    WNBA_2024 = pd.read_csv("2024_WNBA.csv")
    
    # rename columns to match other data frames
    WNBA_2024 = WNBA_2024.rename(columns={"FG%": "FG_Percent", "3P%": "3P_Percent", "FT%": "FT_Percent", "OREB": "ORB", "DREB": "DRB"})
    
    # use same columns for prediction
    WNBA_x = WNBA_2024[["FGA", "FG_Percent", "3PA", "3P_Percent", "FTA", "FT_Percent", "ORB", "DRB", "AST",
                 "STL", "BLK", "TOV", "PF"]]
    WNBA_y = WNBA_2024['WIN%']

    # standardize data
    WNBA_x_scaled = scaler.fit_transform(WNBA_x)

    overallWNBAModel = regression(wNBA_df_config())[2]

    WNBA_2024["Predicted Win%"] = overallWNBAModel.predict(WNBA_x_scaled)

    WNBA_2024_Summary =  WNBA_2024[["TEAM", "WIN%", "Predicted Win%"]]
    
    errorList = []
    for _, row in WNBA_2024_Summary.iterrows():
        error = abs(row['WIN%'] - row["Predicted Win%"]) / row['WIN%']
        errorList.append(error * 100)

    WNBA_2024_Summary["% Error"] = errorList

    return WNBA_2024_Summary


def main():
    print("\n")

    # Create NBA and WNBA data frames
    NBA_df = NBA_df_config()
    wNBA_df = wNBA_df_config()

    NBA_features = regression(NBA_df)[0]
    WNBA_features = regression(wNBA_df)[0]

    r2_Feature_Comparison = NBA_features.merge(WNBA_features, on="Feature", suffixes=("_NBA", "_WNBA"))
    print(r2_Feature_Comparison)
    print("\n")


    NBA_overall = regression(NBA_df)[1]
    WNBA_overall = regression(wNBA_df)[1]

    r2_Overall_Comparison = NBA_overall.merge(WNBA_overall, on="Metric", suffixes=("NBA", "WNBA"))
    print(r2_Overall_Comparison)
    print("\n")

    print(wNBA_2024())


main()


