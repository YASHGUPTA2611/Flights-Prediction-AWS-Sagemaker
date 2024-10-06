import os

import pickle

import warnings

import joblib

import numpy as np

import pandas as pd

import xgboost as xgb

import streamlit as st

import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
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

sklearn.set_config(transform_output="pandas")

# convenience functions


# airline
air_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

#doj
feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]

doj_transformer = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True, format="mixed")),
    ("scaler", MinMaxScaler())
])

# source & destination
location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

def is_north(X):
    columns = X.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai"]
    return (
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

# dep_time & arrival_time
time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scaler", MinMaxScaler())
])

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

time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])

time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2)
])

# duration

def duration_category(X, short=180, med=400):
    return (
        X
        .assign(duration_cat=np.select([X.Duration_in_minute.lt(short),
                                    X.Duration_in_minute.between(short, med, inclusive="left")],
                                    ["short", "medium"],
                                    default="long"))
        .drop(columns="Duration_in_minute")
    )

def is_over(X, value=1000):
    return (
        X
        .assign(**{
            f"duration_over_{value}": X.Duration_in_minute.ge(value).astype(int)
        })
        .drop(columns="Duration_in_minute")
    )

duration_pipe = Pipeline(steps=[
	("cat", FunctionTransformer(func=duration_category)),
	("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])

duration_union = FeatureUnion(transformer_list=[
	("part1", duration_pipe),
	("part2", FunctionTransformer(func=is_over)),
	("part3", StandardScaler())
])

duration_transformer = Pipeline(steps=[
	("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
	("imputer", SimpleImputer(strategy="median")),
	("union", duration_union)
])



# total_stops
def is_direct(X):
    return X.assign(is_direct_flight=X.Total_Stops.eq(0).astype(int))


total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("", FunctionTransformer(func=is_direct))
])


## column transformer
column_transformer = ColumnTransformer(transformers=[
	("air", air_transformer, ["Airline"]),
	("doj", doj_transformer, ["Date_of_Journey"]),
	("location", location_transformer, ["Source", 'Destination']),
    ("time", time_transformer, ["Dep_Time", "Arrival_Time"]),
	("dur", duration_transformer, ["Duration_in_minute"]),
    ("stops", total_stops_transformer, ["Total_Stops"]),],remainder="passthrough" ) 

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
X_train = train.drop(columns="Price")
y_train = train.Price.copy()

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
	options=X_train.Airline.unique()
)

doj = st.date_input("Date of Journey:")

source = st.selectbox(
	"Source",
	options=X_train.Source.unique()
)

destination = st.selectbox(
	"Destination",
	options=X_train.Destination.unique()
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
	Airline=[airline],
	Date_of_Journey=[doj],
	Source=[source],
	Destination=[destination],
	Dep_Time=[dep_time],
	Arrival_Time=[arrival_time],
	Duration_in_minute=[duration],
	Total_Stops=[total_stops],

)).astype({
	col: "str"
	for col in ["Date_of_Journey", "Dep_Time", "Arrival_Time"]
})

if st.button("Predict"):
	saved_preprocessor = joblib.load("preprocessor.joblib")
	x_new_pre = saved_preprocessor.transform(x_new)

	with open("xgboost-model", "rb") as f:
		model = pickle.load(f)
	x_new_xgb = xgb.DMatrix(x_new_pre)
	pred = model.predict(x_new_xgb)[0]

	st.info(f"The predicted price is {pred:,.0f} INR")

