import json
from unidecode import unidecode


def get_request_features(df):
    feature_list = df.columns.tolist()
    request_body_model = {unidecode(key): 1 for key in feature_list}
    return request_body_model
