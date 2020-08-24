import numpy as np

def preprocess(df):
    # df preprocessing for training comes here
    df['quality'] = np.array(df['quality']).astype(np.float32)
    return df
