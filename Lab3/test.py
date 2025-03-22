# import numpy as np
# import pandas as pd
# from keras import optimizers, Sequential
# from keras.src.layers import Dense
# from keras.src.saving import load_model
# from matplotlib import pyplot as plt
# from numpy import mean
# from sklearn.model_selection import RepeatedKFold
#
#
# def get_dataset():
#     # X: samples * inputs
#     # y: samples * outputs
#
#     data = pd.read_csv("data.csv")
#     inputs = data[['Temperature', 'Pressure']].to_numpy()
#     outputs = data[["Thermal conductivity"]].to_numpy()
#     return inputs, outputs
#
#
# # get the model
# def get_model(n_inputs, n_outputs):
#     model = Sequential()
#     model.add(Dense(2, input_dim=n_inputs, activation='sigmoid'))
#     model.add(Dense(n_outputs, activation='linear'))
#     opt1 = optimizers.Adam(learning_rate=0.005)
#     model.compile(loss='mae', metrics=['mape'], optimizer=opt1)
#     model.summary()
#     return model
#
#
# # evaluate a model using repeated k-fold cross-validation
# def evaluate_model(X, y):
#     n_inputs, n_outputs = X.shape[1], y.shape[1]
#     print("Inputs = ", n_inputs, " Outputs = ", n_outputs)
#     # define evaluation procedure
#     cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=22527)
#     # enumerate folds
#     i = 0
#     MAPE = 300
#     ##K-fold
#     for train_ix, test_ix in cv.split(X):
#         # prepare data
#         i = i + 1
#         ##for K-fold
#         X_train, X_test = X[train_ix], X[test_ix]
#         y_train, y_test = y[train_ix], y[test_ix]
#
#         # define model
#         model = get_model(n_inputs, n_outputs)
#         # fit model
#         history = model.fit(X_train, y_train, verbose=0, epochs=1000)
#         plt.plot(history.history['loss'])
#         plt.title('model loss')
#         plt.ylabel('loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'val'], loc='upper left')
#         plt.show()  # thx https://stackoverflow.com/a/56807595
#         # evaluate model on test set
#         [mae_train, mape_train] = model.evaluate(X_train, y_train)
#         [mae_test, mape_test] = model.evaluate(X_test, y_test)
#         [mae, mape] = model.evaluate(X, y)
#         if (mape < MAPE):
#             MAPE = mape
#             model2 = model
#             print("Saving model...")
#         # store result
#         print('fold: %d' % i)
#         print('> MAE train: %.3f' % mae_train)
#         print('> MAE test: %.3f' % mae_test)
#         print('> MAPE train: %.3f' % mape_train)
#         print('> MAPE test: %.3f' % mape_test)
#         print('> MAE total: %.3f' % mae)
#         print('> MAPE total: %.3f' % mape)
#     return mae, mape, model2
#
#
# def normalize_data(X, minsX=None, maxsX=None):
#     nX = X.copy()
#     if minsX is None or maxsX is None:
#         minsX = []
#         maxsX = []
#         for j in range(0, X.shape[1]):
#             minsX.append(min(X[:, j]))
#             maxsX.append(max(X[:, j]))
#
#     for j in range(0, X.shape[1]):
#         for i in range(0, X.shape[0]):
#             if maxsX[j] == minsX[j]:
#                 nX[i, j] = 0.5
#             else:
#                 nX[i, j] = (X[i, j] - minsX[j]) / (maxsX[j] - minsX[j]) * 0.9 + 0.1
#     return nX, minsX, maxsX
#
#
# def denormalize_data(X, minsX, maxsX):
#     dX = X.copy();
#     for j in range(0, X.shape[1]):
#         for i in range(0, X.shape[0]):
#             dX[i, j] = ((X[i, j] - 0.1) / 0.9) * (maxsX[j] - minsX[j]) + minsX[j]
#     return dX
#
#
# # load dataset
# X, y = get_dataset()
# X, minsX, maxsX = normalize_data(X)
# y, minsy, maxsy = normalize_data(y)
#
# # evaluate model
# # mae, mape, model = evaluate_model(X, y)
# # model.save('test_model.keras')
# # print('MAE: %.3f MAPE: %.3f' % (mae, mape))
# model = load_model('test_model.keras')
#
# new_y = model.predict(X)
# dnX = denormalize_data(X, minsX, maxsX)
# dny = denormalize_data(y, minsy, maxsy)
# new_y = denormalize_data(new_y, minsy, maxsy)
#
#
# def mae(y_exp, y_pred):
#     return np.mean([abs(y_exp[i] - y_pred[i]) for i in range(0, y_exp.shape[0])])
#
#
# print('Thermal conductivity MAE ', mae(dny[:, 0], new_y[:, 0]))
#
# plt.figure(figsize=(6, 6))
# plt.plot(dny, new_y, '.', label="Predicted vs Actual")
# plt.plot([min(dny), max(dny)], [min(dny), max(dny)], 'r-', label="Ideal Line")
# plt.xlabel("Actual Thermal conductivity")
# plt.ylabel("Predicted Thermal conductivity")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# P_target = 1.0
# T_target = 320.0
#
# mask = dnX[:, 1] == P_target
# x = dnX[mask]
# y = dny[mask]
# new_y = new_y[mask]
#
# x_target = np.array([[T_target, P_target]])
# x_target_norm, _, _ = normalize_data(x_target, minsX, maxsX)
#
# y_target_norm = model.predict(x_target_norm)
# y_target = denormalize_data(y_target_norm, minsy, maxsy)
#
# plt.figure(figsize=(8, 6))
# plt.scatter(x[:, 0], y[:, 0], label="Экспериментальные данные", color='red')
# plt.plot(x[:, 0], new_y[:, 0], label="Предсказание модели", linestyle='-', color='blue')
# plt.scatter(T_target, y_target, label=f"Предсказанное значение при T={T_target}K", color='green',
#             marker='o', s=100)
#
# plt.title(f"Зависимость теплопроводности от температуры при P={P_target} бар")
# plt.xlabel("Температура, [K]")
# plt.ylabel("Теплопроводность, [мВт/м/К]")
# plt.legend()
# plt.grid(True)
# plt.show()
#
# print(f"Предсказанное значение при T={T_target}K, P={P_target} бар: {y_target[0, 0]:.3f}")
