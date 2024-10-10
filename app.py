import os
import json
import matplotlib.pyplot as plt
import pickle
from functools import lru_cache
import warnings
import catboost as cb
import numpy as np
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.linear_model import LinearRegression
import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
	OneHotEncoder,
	OrdinalEncoder,
	StandardScaler,
	MinMaxScaler,
	PowerTransformer,
	FunctionTransformer
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
	RareLabelEncoder,
	MeanEncoder,
	CountFrequencyEncoder
)

with open("dictionary_distance", 'r') as json_file:
    distance_dictionary = json.load(json_file)
    
sklearn.set_config(transform_output="pandas")
# convenience functions

# airline
# airline_transformer = Pipeline(steps = [
#                                     ("grouper", RareLabelEncoder(tol=  0.1, replace_with = "Other", n_categories=2)),
#                                     ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    
# ])


airline_transformer = Pipeline(steps = [
                                    ("grouper", RareLabelEncoder(tol=  0.05, replace_with = "Other", n_categories=2)),
                                    # ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                                    ("mean_encoder", MeanEncoder()),
                                	("scaler", PowerTransformer())
    
])

# -------------------------------------------------------------------------------------------


# #doj
# feature_to_extract = ["month", "week", "day_of_week" ]

# doj_transformer = Pipeline(steps=[
# 	("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
# 	("scaler", MinMaxScaler())
# ])

feature_to_extract = ["week", "day_of_week"]

doj_transformer = Pipeline(steps=[
	("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
	("scaler", MinMaxScaler())
])



# -------------------------------------------------------------------------------------------

# source & destination

def source_destination(train):
    sd_subset = train[["source","destination"]]
    sd_subset[["source", "destination"]] = sd_subset[["source", "destination"]].apply(lambda col: col.str.lower())
    sd_subset["source_destination"] = (sd_subset["source"].astype(str) + "_" + sd_subset["destination"])
    
    return pd.DataFrame(sd_subset["source_destination"], columns=["source_destination"])


def return_distance_df(train):
    distance = train["source_destination"].map(distance_dictionary)
    
    # Create a DataFrame with the calculated distance
    return pd.DataFrame({"distance_between_cities": distance})

    
## Lets fist convert airline coulumn to Countvectorizer() finction of feature_engine
sd_count_transformer = Pipeline(steps = [
                                    ( "source_destination" , FunctionTransformer(source_destination)),
                                    ("grouper", RareLabelEncoder(tol=  0.1, replace_with = "Other", n_categories=2)),
                                    # ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
                                    ("count_encoder", CountFrequencyEncoder()),
                                	("scaler", PowerTransformer())
    
])

location_pipe2 = Pipeline(steps=[
    ( "source_destination" , FunctionTransformer(source_destination)),
    # ("grouper", RareLabelEncoder(tol=  0.1, replace_with = "Other", n_categories=2)),
	("encoder", FunctionTransformer(func=return_distance_df)),
    ("StandardScaler", StandardScaler())
])


location_transformer = FeatureUnion(transformer_list=[
	("part1", sd_count_transformer),
    ("part2", location_pipe2),
])


# -------------------------------------------------------------------------------------------



# dep_time & arrival_time

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
	columns = X.columns.to_list()
	X_temp = X.assign(**{
		col: pd.to_datetime(X.loc[:, col]).dt.hour
		for col in columns
	})

	return (
		X_temp
		.assign(**{
			f"{col}_part_of_day": np.select(
				[X_temp.loc[:, col].between(morning, noon, inclusive="left"),
				 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
				 X_temp.loc[:, col].between(eve, night, inclusive="left")],
				["morning", "afternoon", "evening"],
				default="night"
			)
			for col in columns
		})
		.drop(columns=columns)
	)


def dep_arrival(train):
    time_subset = train[["dep_time_part_of_day","arrival_time_part_of_day"]]
    time_subset[["dep_time_part_of_day", "arrival_time_part_of_day"]] = time_subset[["dep_time_part_of_day", "arrival_time_part_of_day"]].apply(lambda col: col.str.lower())
    time_subset["dept_arrival"] = (time_subset["dep_time_part_of_day"].astype(str) + "_" + time_subset["arrival_time_part_of_day"])
    
    return pd.DataFrame(time_subset["dept_arrival"], columns=["dept_arrival"])


time_transformer = Pipeline(steps=[
	("part", FunctionTransformer(func=part_of_day)),
	("label_encoder", FunctionTransformer(func=dep_arrival)),
    ("count_encoder", CountFrequencyEncoder()),
    ("scaler", PowerTransformer())
])


# --------------------------------------------------------------------------------------      

# duration    

## Standarize columns=  "total_stops", "duration"

stand = Pipeline(steps=[
    
	("scaling", StandardScaler())
])


# ----------------------------------------------------------------------------------


# column transformer
column_transformer = ColumnTransformer(transformers=[
	("air", airline_transformer, ["airline"]),
	("doj", doj_transformer, ["date_of_journey"]),
	("location", location_transformer, ["source", 'destination']),
    ("time", time_transformer, ["dep_time", "arrival_time"]),
    ("stand", stand, ["duration", "total_stops"])],remainder="passthrough" ) 


# feature selector
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
estimator=estimator,
scoring="r2",
threshold=0.1
) 

# preprocessor
preprocessor = Pipeline(steps=[
("ct", column_transformer),
("selector", selector)
])


# read the training data
train = pd.read_csv("train.csv")
X_train = train.drop(columns="price")
y_train = train.price.copy()

# fit and save the preprocessor
preprocessor.fit(X_train, y_train)
joblib.dump(preprocessor, "preprocessor.joblib")

# web application
st.set_page_config(
	page_title="Flights Prices Prediction",
	page_icon="✈️",
	layout="wide"
)

st.title("Flights Prices Prediction - AWS SageMaker")

# user inputs
airline = st.selectbox(
	"Airline:",
	options=X_train.airline.unique()
)

doj = st.date_input("Date of Journey:")

source = st.selectbox(
	"Source",
	options=X_train.source.unique()
)

destination = st.selectbox(
	"Destination",
	options=X_train.destination.unique()
)

dep_time = st.time_input("Departure Time:")

arrival_time = st.time_input("Arrival Time:")

duration = st.number_input(
	"Duration (mins):",
	step=1
)

total_stops = st.number_input(
	"Total Stops:",
	step=1,
	min_value=0
)


x_new = pd.DataFrame(dict(
	airline=[airline],
	date_of_journey=[doj],
	source=[source],
	destination=[destination],
	dep_time=[dep_time],
	arrival_time=[arrival_time],
	duration=[duration],
	total_stops=[total_stops],

)).astype({
	col: "str"
	for col in ["date_of_journey", "dep_time", "arrival_time"]
})

if st.button("Predict"):
	saved_preprocessor = joblib.load("preprocessor.joblib")
	x_new_pre = saved_preprocessor.transform(x_new)

	with open("xgboost_model.pkl", "rb") as f:
		model = pickle.load(f)
	pred = model.predict(x_new_pre)[0]

	st.info(f"The predicted price is {pred:,.0f} INR")

