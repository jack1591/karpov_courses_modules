# coding: utf-8
import os
import pickle
from catboost import CatBoostClassifier

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("/my/super/catboost_model")
    from_file = CatBoostClassifier()
    model = from_file.load_model(model_path, format = 'cbm')
    return model
