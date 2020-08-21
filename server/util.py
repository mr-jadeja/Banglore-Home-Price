import json,pickle
import numpy as np
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    global __model
    global __data_columns
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    y = np.zeros(len(__data_columns))
    y[0] = sqft
    y[1] = bath
    y[2] = bhk
    if loc_index >= 0:
        y[loc_index] = 1
    return round(__model.predict([y])[0],2)



def load_saved_artifacts():
    print("loading artifacts starting.....")
    global __locations
    global __data_columns
    global __model

    with open("./artifacts/columns.json","r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/banglore_price_model.pickle","rb") as f:
        __model = pickle.load(f)

    print("loading artifacts is done")

def get_location_names():
	load_saved_artifacts()
	return __locations

def get_data_columns():
    return __data_columns


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price(' devarachikkanahalli',1000,2,2))