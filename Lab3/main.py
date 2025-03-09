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


def normalize_data(x):
    x_n = x.copy()

    for j in range(0, x.shape[1]):
        min_val = min(x[:, j])
        max_val = max(x[:, j])

        if max_val == min_val:
            x_n[:, j] = 0.5
        else:
            for i in range(x.shape[0]):
                x_n[i, j] = (x[i, j] - min_val) / (max_val - min_val) * 0.9 + 0.1
    return x_n


def denormalize_data(x_n, min_x, max_x):
    den_x = x_n.copy()
    for j in range(0, x_n.shape[1]):
        for i in range(0, x_n.shape[0]):
            den_x[i, j] = ((x_n[i, j] - 0.1) / 0.9) * (max_x[j] - min_x[j]) + min_x[j]
    return den_x


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


if __name__ == '__main__':
    x, y = get_dataset()
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    y_min = y.min(axis=0)
    y_max = y.max(axis=0)

    x_norm = normalize_data(x)
    y_norm = normalize_data(y)

    model = load_model('lab3MLnew.keras')

    # mae, mape, model = evaluate_model(x_norm, y_norm)
    # model.save('lab3MLnew.keras')
    # print('MAE: %.3f MAPE: %.3f' % (mae, mape))

    new_y = model.predict(x_norm)

    dnX = denormalize_data(y_norm, x_min, x_max)
    dny = denormalize_data(y_norm, y_min, y_max)
    new_y = denormalize_data(new_y, y_min, y_max)


    def mae(y_exp, y_pred):
        return np.mean([abs(y_exp[i] - y_pred[i]) for i in range(0, y_exp.shape[0])])


    print('Thermal conductivity MAE ', mae(dny[:, 0], new_y[:, 0]))

    plt.figure(figsize=(6, 6))
    plt.plot(dny, new_y, '.', label="Predicted vs Actual")
    plt.plot([min(dny), max(dny)], [min(dny), max(dny)], 'r-', label="Ideal Line")
    plt.xlabel("Actual Thermal conductivity")
    plt.ylabel("Predicted Thermal conductivity")
    plt.legend()
    plt.grid(True)
    plt.show()

    P_target = 0.1
    T_target = 320

    x_target = np.array([[T_target, P_target]])

    x_target_norm = x_target.copy()

    for j in range(0, x_target_norm.shape[1]):
        min_val = min(x[:, j])
        max_val = max(x[:, j])

        if max_val == min_val:
            x_target_norm[:, j] = 0.5
        else:
            for i in range(x_target_norm.shape[0]):
                x_target_norm[i, j] = (x_target_norm[i, j] - min_val) / (max_val - min_val) * 0.9 + 0.1

    y_target_norm = model.predict(x_target_norm)
    y_target = denormalize_data(y_target_norm, y_min, y_max)

    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], y[:, 0], label="Экспериментальные данные", color='red')
    plt.plot(x[:, 0], new_y[:, 0], label="Предсказание модели", linestyle='-', color='blue')
    plt.scatter(T_target, y_target, label=f"Предсказанное значение при T={T_target}K", color='green',
                marker='o', s=100)

    plt.title(f"Зависимость теплопроводности от температуры при P={P_target * 10} бар")
    plt.xlabel("Температура, [K]")
    plt.ylabel("Теплопроводность, [мВт/м/К]")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Предсказанное значение при T={T_target}K, P={P_target} бар: {y_target[0, 0]:.3f}")
