from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

import pickle

model_path = "../models/model.pkl"
ohe_path = "../models/ohe.pkl"


def load_model():
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model


def load_encoder():
    with open(ohe_path, 'rb') as file:
        one_hot_enc = pickle.load(file)
    return one_hot_enc