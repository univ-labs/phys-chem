import pandas as pd


def get_dataset():
    # x: samples * inputs
    # y: samples * outputs

    data = pd.read_csv("data.csv")
    inputs = data[['Temperature', 'Pressure']].to_numpy()
    outputs = data[["Thermal conductivity"]].to_numpy()
    return inputs, outputs

