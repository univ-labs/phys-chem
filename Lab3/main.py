import numpy as np
import pandas as pd
from keras import Sequential
from keras.api import optimizers
from keras.src.layers import Dense
from keras.src.saving import load_model
from matplotlib import pyplot as plt
from sklearn.model_selection import RepeatedKFold


def get_dataset():
    # x: samples * inputs
    # y: samples * outputs

    data = pd.read_excel("DataSet.xlsx", engine="openpyxl")
    inputs = data[['Temperature', 'Pressure']][1:].to_numpy(dtype=float)
    outputs = data[["Thermal conductivity"]][1:].to_numpy(dtype=float)
    return inputs, outputs


def normalize_data(X):
    nX = X.copy()
    minsX = []
    maxsX = []
    for j in range(0, X.shape[1]):
        min_val = min(X[:, j])
        max_val = max(X[:, j])
        minsX.append(min_val)
        maxsX.append(max_val)

        if max_val == min_val:
            nX[:, j] = 0.5
        else:
            for i in range(X.shape[0]):
                nX[i, j] = (X[i, j] - min_val) / (max_val - min_val) * 0.9 + 0.1
    return nX, minsX, maxsX


def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(2, input_dim=n_inputs, activation='sigmoid'))
    model.add(Dense(n_outputs, activation='linear'))
    opt1 = optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mae', metrics=['mape'], optimizer=opt1)
    model.summary()
    return model


# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
    n_inputs, n_outputs = X.shape[1], y.shape[1]
    print("Inputs = ", n_inputs, " Outputs = ", n_outputs)
    # define evaluation procedure
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=22527)
    # enumerate folds
    i = 0
    MAPE = 300
    ##K-fold
    for train_ix, test_ix in cv.split(X):
        # prepare data
        i = i + 1
        ##for K-fold
        X_train, X_test = X[train_ix], X[test_ix]
        y_train, y_test = y[train_ix], y[test_ix]

        # define model
        model = get_model(n_inputs, n_outputs)
        # fit model
        history = model.fit(X_train, y_train, verbose=0, epochs=1000)
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()  # thx https://stackoverflow.com/a/56807595
        # evaluate model on test set
        [mae_train, mape_train] = model.evaluate(X_train, y_train)
        [mae_test, mape_test] = model.evaluate(X_test, y_test)
        [mae, mape] = model.evaluate(X, y)
        if (mape < MAPE):
            MAPE = mape
            model2 = model
            print("Saving model...")
        # store result
        print('fold: %d' % i)
        print('> MAE train: %.3f' % mae_train)
        print('> MAE test: %.3f' % mae_test)
        print('> MAPE train: %.3f' % mape_train)
        print('> MAPE test: %.3f' % mape_test)
        print('> MAE total: %.3f' % mae)
        print('> MAPE total: %.3f' % mape)
    return mae, mape, model2


def denormalize_data(x, minsX, maxsX):
    dX = x.copy()
    for j in range(0, x.shape[1]):
        for i in range(0, x.shape[0]):
            dX[i, j] = ((x[i, j] - 0.1) / 0.9) * (maxsX[j] - minsX[j]) + minsX[j]
    return dX


if __name__ == '__main__':
    x, y = get_dataset()
    x, minsX, maxsX = normalize_data(x)
    y, minsY, maxsY = normalize_data(y)

    model = load_model('lab3ML.keras')

    # mae, mape, model = evaluate_model(x, y)
    # model.save('lab3ML.keras')
    # print('MAE: %.3f MAPE: %.3f' % (mae, mape))

    new_y = model.predict(x)

    dnX = denormalize_data(x, minsX, maxsX)
    dny = denormalize_data(y, minsY, maxsY)
    new_y = denormalize_data(new_y, minsY, maxsY)

    def mae(y_exp, y_pred):
        return np.mean([abs(y_exp[i] - y_pred[i]) for i in range(0, y_exp.shape[0])])


    print('Density MAE ', mae(dny[:, 0], new_y[:, 0]))

    plt.figure(figsize=(6, 6))
    plt.plot(dny, new_y, '.', label="Predicted vs Actual")
    plt.plot([min(dny), max(dny)], [min(dny), max(dny)], 'r-', label="Ideal Line")
    plt.xlabel("Actual Thermal conductivity")
    plt.ylabel("Predicted Thermal conductivity")
    plt.legend()
    plt.grid(True)
    plt.show()

    # for P in 100, 50, 10, 1:
    #     i = i + 1
    #     Xa, yn = get_data_at_P(P)
    #     Xn = normalize_data_old_range(Xa, minsX, maxsX)
    #     new_y_at_P = model.predict(Xn)
    #     new_y_at_P = denormalize_data(new_y_at_P, minsY, maxsY)
    #     plt.figure(i)
    #     plt.title("P = %d" % P)
    #     plt.plot(Xa[:, 0], yn[:, 0], c=cmap(i * 2 - 4), label="exp., P = %d" % P)
    #     plt.plot(Xa[:, 0], new_y_at_P[:, 0], c=cmap(i * 2 - 3), label="model, P = % d" % P)
    #     plt.ylabel("Density, kg/m^3")
    #     plt.xlabel("Temperature, K")
