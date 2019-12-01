# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 15:38:44 2019

@author: datacore
"""

# Import general useful packages
import pandas as pd
testing_data = pd.read_csv('D:\\Stock_Prediction\\AutoML_Azure\\test\\2018Q4PredictionTestSet10.csv')

# Get X and y for testing data
y_test = testing_data['ActionTaken']
X_test = testing_data.drop(columns = ['ActionTaken', 'ClassInd'])
print(X_test.head())

#example: scikit-learn and Swagger
import json
import numpy as np
from sklearn.externals import joblib
#from sklearn.linear_model import Ridge
from azureml.core.model import Model

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

#filename = "C:/Users/datacore/OneDrive/Desktop/Capstone Project/model.pkl"
# deserialize the model file back into a sklearn model
#import automl
import azureml.train.automl
model_path = "D:\\Stock_Prediction\\AutoML_Azure\\new_model2.pkl"
# deserialize the model file back into a sklearn model
model = joblib.load(model_path)

prediction = model.predict(X_test)

predictions_df = pd.DataFrame(prediction)
predictions_df.rename(columns={0:'ScoredLabel'}, inplace=True)

result1 = pd.concat([testing_data, predictions_df], axis=1)
print(result1)
result1.to_csv('D:\\Stock_Prediction\\AutoML_Azure\\test\\result2.csv', index=False)