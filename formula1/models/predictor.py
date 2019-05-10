import pickle
import numpy as np
import pandas as pd

def main():
    fitted_model = pickle.load(open('model.pkl', 'rb'))

    return fitted_model

    # what will be the unseen data? based on drivername and get from DB?
