import collections


from networkx import degree_centrality, in_degree_centrality, closeness_centrality

import math
import random
import time

import networkx
import numpy
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from networkx import degree_centrality, in_degree_centrality

import random
import pickle
from sklearn.model_selection import train_test_split

import collections
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

import statistics

from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.metrics import ACC
from keras.metrics import accuracy
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats


# ********** functions **********

def draw_graph(_graph, _seed):
    pos = nx.spring_layout(_graph, seed=5)
    nx.draw_networkx_nodes(
        _graph, pos, linewidths=1, node_size=200, node_color='pink', alpha=1,
    )

    nx.draw_networkx_edges(_graph, pos, width=1)
    nx.draw_networkx_labels(_graph, pos, font_size=10, font_family="sans-serif")
    ax = plt.gca()
    ax.margins(0)
    plt.axis("tight")
    plt.tight_layout()
    plt.show()


def draw_graph_with_label(_graph, _seed):
    pos = nx.spring_layout(_graph, seed=_seed)
    nx.draw_networkx_nodes(
        _graph, pos, linewidths=1, node_size=100, node_color='pink', alpha=1,
    )
    nx.draw_networkx_edges(_graph, pos, width=1)
    nx.draw_networkx_labels(_graph, pos, font_size=10, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(_graph, "weight")
    nx.draw_networkx_edge_labels(_graph, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# machine learning method


def ml_prediction(node):
    pr_df = pd.DataFrame()
    pr_df['node'] = workload_df[node]
    pr_df['past1'] = workload_df[node].shift(+1)
    pr_df['past2'] = workload_df[node].shift(+2)
    pr_df['past3'] = workload_df[node].shift(+3)
    pr_df['past4'] = workload_df[node].shift(+4)
    pr_df['past5'] = workload_df[node].shift(+5)
    pr_df['past6'] = workload_df[node].shift(+6)
    pr_df['past7'] = workload_df[node].shift(+7)
    pr_df['past8'] = workload_df[node].shift(+8)
    pr_df['past9'] = workload_df[node].shift(+9)
    pr_df['past10'] = workload_df[node].shift(+10)
    pr_df['past11'] = workload_df[node].shift(+11)
    pr_df['past12'] = workload_df[node].shift(+12)
    pr_df['past13'] = workload_df[node].shift(+13)
    pr_df['past14'] = workload_df[node].shift(+14)
    pr_df = pr_df.dropna()

    # pr_df.to_csv("mini graph/pr_df.csv")
    knn = KNeighborsRegressor(n_neighbors=2)
    lin_model = LinearRegression()
    lasso_model = Lasso(alpha=10)
    ridge_model = Ridge(alpha=1, random_state=1)
    # en_model = ElasticNet(alpha=0.5, l1_ratio=0.4, max_iter=10000000)
    # tree_model = DecisionTreeRegressor()
    # rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    lassocv_model = LassoCV(cv=5)
    poly_model = PolynomialFeatures(degree=2, include_bias=False)
    poly_reg_model = LinearRegression()
    # xgboot_model = GradientBoostingRegressor(max_depth=4, learning_rate=0.1)
    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1.5, 10], 'gamma': [1e-7, 1e-4],
                  'epsilon': [0.1, 0.2, 0.5, 0.3]}
    # SVR_model = SVR(C=1500000, epsilon=0.1)
    # from sklearn.model_selection import GridSearchCV
    # clf = GridSearchCV(SVR_model, parameters)

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, y = pr_df['past1'], pr_df['past2'], pr_df['past3'], \
                                                                     pr_df[
                                                                         'past4'], pr_df['past5'], pr_df['past6'], \
                                                                     pr_df['past7'], pr_df['past8'], pr_df['past9'], \
                                                                     pr_df['past10'], \
                                                                     pr_df['past11'], pr_df['past12'], pr_df['past13'], \
                                                                     pr_df['past14'], pr_df['node']
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, y = np.array(x1), np.array(x2), np.array(x3), np.array(
        x4), np.array(x5), np.array(x6), np.array(x7), np.array(x8), np.array(x9), np.array(x10), np.array(x11), \
                                                                     np.array(x12), np.array(x13), np.array(
        x14), np.array(y)
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, y = x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(
        -1, 1), \
                                                                     x4.reshape(-1, 1), x5.reshape(-1, 1), x6.reshape(
        -1, 1), x7.reshape(-1, 1), x8.reshape(-1, 1), \
                                                                     x9.reshape(-1, 1), x10.reshape(-1, 1), x11.reshape(
        -1, 1), x12.reshape(-1, 1), x13.reshape(-1, 1), x13.reshape(-1, 1) \
        , y.reshape(-1, 1)

    final_x = np.concatenate((x1, x2, x3, x4, x5), axis=1)

    # 80% data for train and 20% for test
    train_size = int(len(final_x) * 0.8)
    xtrain, xtest, ytrain, ytest = final_x[:train_size], final_x[train_size:], \
                                   y[:train_size], y[train_size:]

    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # scores = cross_val_score(lin_model, final_x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    # print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    print("final x: ", len(final_x))
    print("train size: ", len(xtrain))
    print("test size: ", len(xtest))
    knn.fit(xtrain, ytrain)
    lin_model.fit(xtrain, ytrain)
    lasso_model.fit(xtrain, ytrain)
    ridge_model.fit(xtrain, ytrain)
    # en_model.fit(xtrain, ytrain)
    # tree_model.fit(xtrain, ytrain)
    # rf_model.fit(xtrain, ytrain.ravel())
    lassocv_model.fit(xtrain, ytrain)
    poly_model.fit(poly_model.fit_transform(xtrain), ytrain)
    poly_reg_model.fit(poly_model.fit_transform(xtrain), ytrain)
    # xgboot_model.fit(xtrain, ytrain)
    # SVR_model.fit(xtrain, ytrain)
    # clf.fit(xtrain, ytrain)
    # print("best params for SVR:", clf.best_params_)

    print("Training set score: {:.2f}".format(lasso_model.score(xtrain, ytrain)))
    print("Test set score: {:.2f}".format(lasso_model.score(xtest, ytest)))

    knn_pred = knn.predict(xtest)
    lr_pred = lin_model.predict(xtest)
    lasso_pred = lasso_model.predict(xtest)
    ridge_pred = ridge_model.predict(xtest)
    # elastic_pred = en_model.predict(xtest)
    # tree_pred = tree_model.predict(xtest)
    # rf_pred = rf_model.predict(xtest)
    lassocv_pred = lassocv_model.predict(xtest)
    # poly_pred = poly_reg_model.predict(xtest)
    # xgboot_pred = xgboot_model.predict(xtest)
    # SVR_pred = SVR_model.predict(xtest)

    if node == '55':
        # plt.subplot(11, 10, list(DAG2.nodes).index(int(node))+1)
        plt.title(node, fontsize=8, color='green', pad=0.1)
        plt.plot(ytest[-100:], label="actual", color='blue')
        plt.plot(lr_pred[-100:], label="LR", color='red')
        # plt.xticks(fontsize=5)
        # plt.yticks(fontsize=5)
        # plt.plot(lasso_pred[-100:], label='Lasso')
        # plt.plot(ridge_pred[-100:], label='ridge')
        # plt.plot(elastic_pred[-100:], label='EN')
        # plt.plot(tree_pred[-100:], label='tree')
        # plt.plot(rf_pred[-100:], label="RF")
        plt.plot(knn_pred[-100:], label='knn')
        plt.plot(poly_reg_model.predict(poly_model.fit_transform(xtest))[-100:], label='poly')
        # plt.plot(SVR_pred, label='SVR')
        plt.legend()
        plt.show()

    knn_rmse = sqrt(mean_squared_error(ytest[-371:, ], knn_pred[-371:, ]))
    print("KNN ML MSE: ", knn_rmse, "for node: ", node)

    lr_rmse = sqrt(mean_squared_error(ytest[-371:, ], ridge_pred[-371:, ]))
    print("LR ML MSE: ", lr_rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-371:, ], lasso_pred[-371:, ]))
    print("Lasso ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-371:, ], ridge_pred[-371:, ]))
    print("Ridge ML MSE: ", rmse, "for node: ", node)

    # rmse = sqrt(mean_squared_error(ytest[-371:, ], tree_pred[-371:, ]))
    # print("tree ML MSE: ", rmse, "for node: ", node)
    #
    # rmse = sqrt(mean_squared_error(ytest[-371:, ], elastic_pred[-371:, ]))
    # print("EN ML MSE: ", rmse, "for node: ", node)
    #
    # rmse = sqrt(mean_squared_error(ytest[-371:, ], rf_pred[-371:, ]))
    # print("RF ML MSE: ", rmse, "for node: ", node)
    #
    # rmse = sqrt(mean_squared_error(ytest[-371:, ], xgboot_pred[-371:, ]))
    # print("XGboot ML MSE: ", rmse, "for node: ", node)
    #
    # rmse = sqrt(mean_squared_error(ytest[-371:, ], SVR_pred[-371:, ]))
    # print("SVR ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest, lassocv_pred))
    print("Lasso cv ML MSE: ", rmse)

    rmse = sqrt(mean_squared_error(ytest, poly_reg_model.predict(poly_model.fit_transform(xtest))))
    print("Polynomial ML MSE: ", rmse, "for node: ", node)

    predict = []
    for x in ridge_pred.tolist():
        predict.append(round(x[0], 4))

    x1_pred = pr_df.iloc[-1, 0]
    x2_pred = pr_df.iloc[-2, 0]
    x3_pred = pr_df.iloc[-3, 0]
    x4_pred = pr_df.iloc[-4, 0]
    x5_pred = pr_df.iloc[-5, 0]

    x1_pred, x2_pred, x3_pred, x4_pred, x5_pred = np.array(x1_pred), np.array(x2_pred), np.array(x3_pred), np.array(
        x4_pred), np.array(x5_pred)
    x1_pred, x2_pred, x3_pred, x4_pred, x5_pred = x1_pred.reshape(-1, 1), x2_pred.reshape(-1, 1), x3_pred.reshape(-1,
                                                                                                                  1), x4_pred.reshape(
        -1, 1), x5_pred.reshape(-1, 1)

    # print(x1_pred, ", ", x2_pred, ", ", x3_pred)
    final_pred = np.concatenate((x1_pred, x2_pred, x3_pred, x4_pred, x5_pred), axis=1)
    # prediction_res = lin_model.predict(final_pred)
    # print("node ", node, ": ", prediction_res)
    return predict, lr_rmse


def df_to_x_y(df, window_size=9):
    # df_as_np = df.to_numpy()
    x = []
    y = []
    for i in range(len(df) - window_size):
        row = [[a] for a in df[i: i + 9]]
        x.append(row)
        label = df[i + 9]
        y.append(label)
    return numpy.array(x), numpy.array(y)


def rnn_prediction(node):
    tf.random.set_seed(42)
    np.random.seed(37)
    data = workload_df[node]
    # data = data.sample(frac=1)
    dataset = data.values
    # dataset = dataset.astype('float32')

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    print("dataset shape", dataset.shape)
    train_size = int(len(dataset) * 0.8)

    val_Size = int(len(dataset) * 0.2) + train_size
    # test_size = len(dataset) - train_size - val_Size
    test_size = 20
    train, val, test = dataset[:train_size, :], dataset[train_size:, :], dataset[train_size:, :]
    window_size = 9
    data2_x, data2_y = df_to_x_y(dataset, window_size)
    x2_train, x2_test, y2_train, y2_test = train_test_split(data2_x, data2_y, train_size=0.8, shuffle=False,
                                                            random_state=42)

    x_train, y_train = df_to_x_y(train, window_size)
    x_val, y_val = df_to_x_y(val, window_size)
    x_test, y_test = df_to_x_y(test, window_size)

    from numpy.random import seed
    # seed(100)
    model1 = Sequential()
    model1.add(InputLayer((9, 1)))
    model1.add(LSTM(100, batch_size=64))
    # model1.add(Dropout(0.2))
    # model1.add(Dropout(0.15))
    # model1.add(Dense(100, 'relu'))
    # model1.add(Dropout(0.1))
    # model1.add(Dense(32, 'relu'))
    # model1.add(Dropout(0.25))
    model1.add(Dense(1, 'relu'))
    # model1.save('mini graph/mymodel')
    # from keras.models import load_model
    #
    # model1 = load_model('mini graph/mymodel')
    model1.summary()

    cp = ModelCheckpoint('model1/', save_best_only=True)
    save_weights_at = 'basic_lstm_model'
    save_best = ModelCheckpoint(save_weights_at, monitor='val_loss',
                                save_best_only=True, save_weights_only=False, mode='min',
                                period=1)
    model1.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])
    # hist = model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, callbacks=[cp])
    lr_scheduler = LearningRateScheduler(scheduler)
    stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=50, min_lr=0)
    hist = model1.fit(x_train, y_train, validation_split=0.2, epochs=1200,
                      callbacks=[save_best, stopping, lr_scheduler])

    # from keras.models import load_model
    # model1 = load_model('model1/')

    train_predictions = model1.predict(x_train).flatten()
    train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

    train_results = pd.DataFrame(data={'train predictions': list(train_predictions), 'actual': list(y_train)})
    train_results.to_csv("mini graph/train.csv")
    plt.plot(train_results.iloc[:, 0], color='red')
    plt.plot(train_results.iloc[:, 1], color='blue')
    plt.title("train")
    plt.show()

    val_predictions = model1.predict(x_val).flatten()
    val_predictions = scaler.inverse_transform(val_predictions.reshape(-1, 1))
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))

    val_reults = pd.DataFrame(data={'val predictions': list(val_predictions), 'actual': list(y_val)})

    plt.plot(val_reults['val predictions'], color='red')
    plt.plot(val_reults['actual'], color='blue')
    plt.title('val')
    plt.show()
    print("RNN evaluation", model1.evaluate(x_test, y_test))

    test_predictions = model1.predict(x_test).flatten()
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    # x_test = scaler.inverse_transform(x_test.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_results = pd.DataFrame(data={'test predictions': list(test_predictions), 'actual': list(y_test)})
    test_results.to_csv("mini graph/test.csv")

    print("rnn size:", len(test_predictions), len(y_test), len(x2_test), len(y2_test))
    print("rnn test pred:", test_predictions)
    plt.plot(test_results['test predictions'], color='red')
    plt.plot(test_results['actual'], color='blue')
    plt.title('test')
    plt.show()

    print("MSE for train RNN: ", sqrt(mean_squared_error(y_train, train_predictions)))
    # print("MSE for val RNN: ", sqrt(mean_squared_error(y_val, val_predictions)))
    print("MSE for test RNN: ", sqrt(mean_squared_error(y_test, test_predictions)))

    plt.plot(hist.history['loss'], label='Train')
    plt.plot(hist.history['val_loss'], label='validation')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('mean absolute error')
    plt.legend()
    plt.show()

    plt.plot(hist.history['lr'], label='learning rate')
    plt.title('learning rate')
    plt.show()

    print(hist.history.keys())
    print(hist.history['mean_squared_error'])
    plt.plot(hist.history['mean_squared_error'])
    plt.plot(hist.history['val_mean_squared_error'])
    plt.title("Model's Training & Validation MSE across epochs")
    plt.ylabel('mean squared error')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    evaluation = model1.evaluate(x_test, y_test)
    for i in range(len(model1.metrics_names)):
        print("metric:", model1.metrics_names[i], " : ", evaluation[i])

    # return test_predictions.reshape(-1)
    return test_results


def scheduler(epoch, lr):
    if epoch < 50:
        return 0.01
    if epoch < 350:
        return 0.001
    else:
        return 0.0001

# ********** construct base graph **********

G = nx.DiGraph()
G.add_nodes_from(range(0, 11))
G.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (4, 3), (5, 2), (5, 4), (5, 6), (5, 8), (5, 9),
                  (7, 10), (8, 7), (9, 6), (9, 8), (9, 11)])

draw_graph(G, 5)


pickle.dump(G, open('mini_graph.pickle', 'wb'))
# nx.write_gexf(G, "mini_graph.gexf")

# load graph object from file
# G = pickle.load(open('mini_graph.pickle', 'rb'))

centrality = degree_centrality(G)
print("centralities of G2: ", centrality)
print("in degree centrality: ", in_degree_centrality(G))
print(list(G.out_edges(5)))
print(list(G.out_edges(5))[1][1])
print(list(G.neighbors(5)))
betweenness = nx.betweenness_centrality(G, k=10, normalized=True, endpoints=True)
print("betweenness of G nodes: ", betweenness)

close_centrality = closeness_centrality(G)
centrality_list = []
for node in G:
    centrality_list.append(betweenness[node])
centrality_list.sort()
print(centrality_list)
centrality_list = list(dict.fromkeys(centrality_list))
for node in G:
    if betweenness[node] == centrality_list[-1]:
        print("1 maximum degree centrality in G is ", centrality_list[-1], "\nand it belongs to ", node)
    elif betweenness[node] == centrality_list[-2]:
        print("2 maximum degree centrality in G is ", centrality_list[-2], "\nand it belongs to ", node)
    elif betweenness[node] == centrality_list[-3]:
        print("3 maximum degree centrality in G is ", centrality_list[-3], "\nand it belongs to ", node)

# ********** Generate dataset for edges **********

# edge_list = []
# indx = 0
# dataframe = pd.DataFrame(columns=list(G.edges))
#
# for edge in G.edges:
#     print("edge 0: ", edge[0])
#     if edge[0] == 0:
#         data = [random.choice([520, 530, 550])]
#     elif edge[0] == 1:
#         data = [random.choice([200, 205])]
#     elif edge[0] == 5:
#         data = [random.choice([151, 152, 153, 154])]
#     elif edge[0] == 4:
#         data = [random.choice([142, 143, 144, 145])]
#     elif edge[0] == 9:
#         data = [random.choice([133, 134, 136])]
#     elif edge[0] == 8:
#         data = [random.choice([125, 126, 127])]
#     elif edge[0] == 7:
#         data = [random.choice([115, 116, 118])]
#     elif edge[0] in [2, 3, 6, 10, 11]:
#         data = [random.choice([105, 106, 108])]
#
#     edge_list.append(data)
#     indx += 1
#
# print("edge list length: ", len(edge_list))
#
# edges11 = []
# nodes11 = []
# for path in nx.all_simple_edge_paths(G, 0, 11):
#     print(path)
#     for x in path:
#         edges11.append(x)
#         for y in x:
#             nodes11.append(y)
#
#
# timeout = time.time() + 60 * 5 # 15 minutes from now
# i = 0
# while time.time() < timeout:
#     for li in edge_list:
#         index = edge_list.index(li)
#         if list(G.edges)[index] in edges11:
#             if time.time() < timeout - 280:
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.3, 0.5])
#             elif (time.time() > timeout - 280) and (time.time() < timeout - 220):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 220) and (time.time() < timeout - 210):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 210) and (time.time() < timeout - 160):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 160) and (time.time() < timeout - 150):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 150) and (time.time() < timeout - 100):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 100) and (time.time() < timeout - 90):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 90) and (time.time() < timeout - 60):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 60) and (time.time() < timeout - 50):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 50) and (time.time() < timeout - 5):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif time.time() > timeout - 5:
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#
#         else:
#             x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.4, 0.35])
#             if time.time() > timeout - 250:
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
#         li.append(x)
#         edge_list[index] = li
#     i += 1
#     time.sleep(0.01)
#
#
# # while time.time() < timeout:
# #     for li in edge_list:
# #         index = edge_list.index(li)
# #         x = li[i] + numpy.random.choice([-2, -1, 0, 1, 2], p=[0.2, 0.2, 0.2, 0.2, 0.2])
# #         li.append(x)
# #         edge_list[index] = li
# #     i += 1
#
#
# for (li, edge) in zip(edge_list, G.edges):
#     dataframe[edge] = li
#
#
# dataframe.to_csv("mini graph/mini graph edges.csv")
# print("done!!!!!!")
dataframe = pd.read_csv("mini graph/mini graph edges.csv", index_col=0)
dataframe.plot()
plt.show()

API = dataframe.iloc[:, 0]

# ********** aggregation data to graphs ***********

# graph_list = []
# for i in range(len(dataframe)):
#     graph_list.append(nx.Graph())
#     graph_list[i] = G.copy()
#
# j = 0
# for ind in dataframe.index:
#     for edge in graph_list[j].edges:
#         (graph_list[j])[edge[0]][edge[1]]["weight"] = dataframe[str(edge)][ind]
#     j += 1

# ********** change to probabilistic graphs **********

# overal_list = []
# for graph in graph_list:
#     sum_list = []
#     for node in graph.nodes():
#         result = 0
#         for edge in list(graph.in_edges(node)):
#             result += graph[edge[0]][edge[1]]["weight"]
#         sum_list.append(result)
#     overal_list.append(sum_list)
#
# lists = {}
# for graph in graph_list:
#     for node in graph.nodes():
#         if node == 0:
#             for edge in list(graph.out_edges(node)):
#                 graph[edge[0]][edge[1]]["weight"] /= API[graph_list.index(graph)]
#                 if str(edge) not in lists.keys():
#                     lists[str(edge)] = []
#                 lists[str(edge)].append(graph[edge[0]][edge[1]]["weight"])
#         else:
#             for edge in list(graph.out_edges(node)):
#                 graph[edge[0]][edge[1]]["weight"] /= (overal_list[graph_list.index(graph)])[
#                     list(graph.nodes()).index(node)]
#                 if str(edge) not in lists.keys():
#                     lists[str(edge)] = []
#                 lists[str(edge)].append(graph[edge[0]][edge[1]]["weight"])
#
# # edge_df = pd.DataFrame()
# # for key in lists:
# #     edge_df[key] = lists[key].copy()
# edge_df = pd.DataFrame.from_dict(lists)
# edge_df.to_csv('mini graph/p_edge.csv')
# print("done1")

edge_df = pd.read_csv('mini graph/p_edge.csv', index_col=0)

# ********** calculate probability for each node **********

# node_df = pd.DataFrame(columns=list(G.nodes))
#
# lists1 = []
# number = len(edge_df)
# for graph in graph_list:
#     lists = []
#     for node in graph.nodes():
#         result = 0
#         for path in nx.all_simple_paths(graph, source=0, target=node):
#             res = 1
#             for i in range(len(path) - 1):
#                 res *= float(graph[path[i]][path[i + 1]]["weight"])
#             result += res
#         lists.append(result)
#         # node_df.loc[graph_list.index(graph), node] = float(format(result, '.3f'))
#     number -= 1
#     print("nemberr:", number)
#     lists1.append(lists)
#
# num = 0
# for li in lists1:
#     node_df.loc[len(node_df)] = li
#     num += 1
#     print("num=", num)
# node_df.to_csv("mini graph/p_node.csv")
# print("done2!")

node_df = pd.read_csv("mini graph/p_node.csv", index_col=0)

# ********** convert multiple graphs to one PHFS graph & calculate function scores **********

# phfs_df = pd.DataFrame()
# fs_df = pd.DataFrame()
# for col in node_df.columns:
#     for j in range(1, int(len(node_df) / 10) + 1):
#         myList = []
#         f_score = 0
#         for i in range((j - 1) * 10, j * 10):
#             myList.append(node_df[col][i])
#         collection = dict(collections.Counter(myList))
#         for key in collection:
#             collection[key] /= 10
#         phfs_df.loc[j - 1, col] = [collection]
#         for key in collection:
#             f_score += float(key) * collection[key]
#         fs_df.loc[j - 1, str(col) + ""] = float(format(f_score, '.4f'))
#
#     # if len(node_df) % 10 > 0:
#     #     myList = []
#     #     f_score = 0
#     #     for i in range(j * 10, j * 10 + (len(node_df) % 10)):
#     #         print("j=", j, "i=", i)
#     #         myList.append(node_df[col][i])
#     #     collection = dict(collections.Counter(myList))
#     #     print(collection)
#     #     for key in collection:
#     #         collection[key] /= 10
#     #     print("hi", collection)
#     #     phfs_df.loc[j, col] = [collection]
#     #     for key in collection:
#     #         f_score += float(key) * collection[key]
#     #     fs_df.loc[j, col] = float(format(f_score, '.3f'))
#
# phfs_df.to_csv("mini graph/phfs.csv")
# fs_df.to_csv("mini graph/function_scores.csv")

phfs_df = pd.read_csv("mini graph/phfs.csv", index_col=0)
fs_df = pd.read_csv("mini graph/function_scores.csv", index_col=0)
fs_df.drop("0", inplace=True, axis=1)

API2 = []
for i in range(int(len(API) / 10)):
    x = 0
    for j in range(10):
        x += API[i * 10 + j]
    API2.append(x / 10)

# ********** prediction  probabilities for each node **********

plt.suptitle("function scores for each node:")
for i in range(1, 12):
    plt.subplot(3, 4, i)
    plt.title(i)
    plt.plot(fs_df[str(i)])

plt.tight_layout()
plt.show()

# ********** decision making table **********
print("API: ", API)
print("API2: ", API2)
workload_df = fs_df.copy()
workload_df.loc[:, "API"] = API2
workload_df.loc[:, workload_df.columns != "API"] = \
    round(workload_df.loc[:, workload_df.columns != "API"].multiply(workload_df["API"], axis="index"), 4)
# workload_df['0'] = workload_df["API"]
workload_df.drop("API", inplace=True, axis=1)

workload_df.to_csv("mini graph/workloads.csv")

predictions = []
plt.suptitle("function scores and predictions:")

for node in range(1, 12):
    # predictions.append(round(ml_prediction(str(node)).tolist()[0][0], 4))
    predictions.append(ml_prediction(str(node))[len(ml_prediction(str(node))) - 1])

plt.tight_layout(pad=0.25)
plt.legend(loc=0, bbox_to_anchor=(1.5, 1.5, 0.4, -0.4))
plt.show()

UCPU_df = workload_df.copy()
Umem_df = workload_df.copy()
for column in UCPU_df:
    if int(column) % 2 == 0:
        UCPU_df[column] = round(UCPU_df[column] * 5, 4)  # 4.2
        Umem_df[column] = round(Umem_df[column] * 0.62, 4)  # 0.01
    else:
        UCPU_df[column] = round(UCPU_df[column] * 1.2, 4)  # 0.01
        Umem_df[column] = round(Umem_df[column] * 20, 4)  # 4.88

UCPU_df.to_csv("mini graph/CPU util.csv")
Umem_df.to_csv("mini graph/memory util.csv")


# ********** decision making table **********

dm_df = pd.DataFrame(columns=fs_df.columns, index=["w(t+1)", "U(CPU)", "U(mem)"])
predictions2 = pd.read_csv("mini graph/RNN.csv", index_col=0)

dm_df.loc["w(t+1)"] = predictions2.iloc[-371, :]
dm_df.loc["U(CPU)"] = UCPU_df.loc[len(UCPU_df) - 372, :]
dm_df.loc["U(mem)"] = Umem_df.loc[len(Umem_df) - 372, :]

dm_df.to_csv("mini graph/predicted.csv")

UCPU_pred = []
Umem_pred = []
for column in dm_df:
    if int(column) % 2 == 0:
        UCPU_pred.append(round(dm_df.loc["w(t+1)", column] * 5, 4))  # 4.2
        Umem_pred.append(round(dm_df.loc["w(t+1)", column] * 0.62, 4))  # 0.01
    else:
        UCPU_pred.append(round(dm_df.loc["w(t+1)", column] * 1.2, 4))  # 0.01
        Umem_pred.append((round(dm_df.loc["w(t+1)", column] * 20, 4)))  # 4.88

print("U CPU pred: ", UCPU_pred)
print("U mem pred:", Umem_pred)


# ********** resources **********

def container(index):
    node = str(index)
    cpu = dm_df.loc['U(CPU)', node]
    mem = dm_df.loc['U(mem)', node]
    print("cpu in dm df: ", cpu, node)
    if int(node) % 2 == 0:
        max_cpu = cpu / 0.9
        n = int(max_cpu / 200) + 1
        max_cpu = n * 200
        if cpu / max_cpu > 0.9:
            n += 1
        maximum = (n * 200)
    else:
        max_mem = mem / 0.9
        n = int(max_mem / 1000) + 1
        max_mem = n * 1000
        if mem / max_mem > 0.9:
            n += 1
        maximum = (n * 1000)
    return maximum


def container2(index):
    node = str(index)
    cpu = dm_df.loc['U(CPU)', node]
    mem = dm_df.loc['U(mem)', node]
    if int(node) % 2 == 0:
        max_mem = mem / 0.9
        n = int(max_mem / 25) + 1
        max_mem = n * 25
        if mem / max_mem > 0.9:
            n += 1
        maximum = (n * 25)
    else:
        max_cpu = cpu / 0.9
        n = int(max_cpu / 60) + 1
        max_cpu = n * 60
        if cpu / max_cpu > 0.9:
            n += 1
        maximum = (n * 60)
    return maximum


# ********** decision making **********

# normalization

wt = workload_df.iloc[-2, :]
wt1 = dm_df.loc['w(t+1)']
dif_df = pd.DataFrame(columns=dm_df.columns)

# normalization
wt = workload_df.iloc[-372, :]
# wt = predictions2.iloc[-2, :]
wt1 = dm_df.loc['w(t+1)']
dif_df = pd.DataFrame(columns=dm_df.columns)

conts = []
for col in dm_df:
    conts.append(container(int(col)))

conts2 = []
for col in dm_df:
    conts2.append(container2(int(col)))

divided2 = []
divided3 = []
for col in dm_df:
    if int(col) % 2 == 0:
        n = max(conts[int(col) - 1]/200, conts2[int(col) - 1]/25)
        # print("nodeee", col, conts[list(G.nodes).index(int(col))]/200, conts2[list(G.nodes).index(int(col))]/25)
        divided2.append(dm_df.loc['U(CPU)', col] / (n*200) * 100)
        print(dm_df.loc['U(CPU)', col], n*200)
        divided3.append(dm_df.loc['U(mem)', col] / (n*25) * 100)

    else:
        n = max(conts[int(col)-1]/1000, conts2[int(col)-1]/60)
        divided2.append(dm_df.loc['U(CPU)', col] / (n*60) * 100)
        divided3.append(dm_df.loc['U(mem)', col] / (n*1000) * 100)

ucpu = dm_df.loc['U(CPU)']
umem = dm_df.loc['U(mem)']


dif_df.loc['w(t+1) - w(t)'] = wt1 - wt
dif_df.loc['ucpu(t)'] = divided2
dif_df.loc['umem(t)'] = divided3

dif_df = dif_df.round(4)
dif_df.to_csv("mini graph/differenced_dm.csv")

divisors = []
for j in range(len(dif_df)):
    row = np.array(dif_df.iloc[j, :])
    divisors.append(np.sqrt(row @ row))

for col in dif_df.columns:
    dif_df.loc['w(t+1) - w(t)', col] /= divisors[0]
    dif_df.loc['ucpu(t)', col] /= divisors[1]
    dif_df.loc['umem(t)', col] /= divisors[2]
    dif_df.loc['umem(t)', col] /= divisors[2]

weights1 = [0.45, 0.45, 0.1]
weights2 = [0.45, 0.1, 0.45]

for col in dif_df.columns:
    if int(col) % 2 == 0:
        dif_df.loc['w(t+1) - w(t)', col] *= weights1[0]
        dif_df.loc['ucpu(t)', col] *= weights1[1]
        dif_df.loc['umem(t)', col] *= weights1[2]
    else:
        dif_df.loc['w(t+1) - w(t)', col] *= weights2[0]
        dif_df.loc['ucpu(t)', col] *= weights2[1]
        dif_df.loc['umem(t)', col] *= weights2[2]

a_pos = []
a_neg = []
for j in range(len(dif_df)):
    row = dif_df.iloc[j, :]
    max_val = np.max(row)
    min_val = np.min(row)

    a_pos.append(max_val)
    a_neg.append(min_val)

sp = []
sn = []
cs = []
for col in dif_df.columns:
    diff_pos = pow((dif_df.loc['w(t+1) - w(t)', col] - a_pos[0]), 2)
    diff_pos += pow(dif_df.loc['ucpu(t)', col] - a_pos[1], 2)
    diff_pos += pow(dif_df.loc['umem(t)', col] - a_pos[2], 2)
    diff_pos = sqrt(diff_pos)

    diff_neg = pow(dif_df.loc['w(t+1) - w(t)', col] - a_neg[0], 2)
    diff_neg += pow(dif_df.loc['ucpu(t)', col] - a_neg[1], 2)
    diff_neg += pow(dif_df.loc['umem(t)', col] - a_neg[2], 2)
    diff_neg = sqrt(diff_neg)

    sp.append(diff_pos)
    sn.append(diff_neg)
    cs.append(sn[-1] / (sp[-1] + sn[-1]))

dif_df.loc["pos_dist"] = sp
dif_df.loc["neg_dist"] = sn
dif_df.loc["score"] = cs

dif_df.to_csv("mini graph/normalized.csv")

for row in dif_df.index:
    print(f'Max element of row {row} is:', max(dif_df.loc[row]), ' and its belong to ', )

cs.sort(reverse=True)

for col in dif_df.columns:
    if dif_df.loc["score", col] == cs[0]:
        print("first max for column: ", col, " is: ", cs[0])
    if dif_df.loc["score", col] == cs[1]:
        print("second max for column: ", col, " is: ", cs[1])
    if dif_df.loc["score", col] == cs[2]:
        print("third max for column: ", col, " is: ", cs[2])
    if dif_df.loc["score", col] == cs[3]:
        print("forth max for column: ", col, " is: ", cs[3])
    if dif_df.loc["score", col] == cs[4]:
        print("5th max for column: ", col, " is: ", cs[4])
    if dif_df.loc["score", col] == cs[5]:
        print("6th max for column: ", col, " is: ", cs[5])
    if dif_df.loc["score", col] == cs[6]:
        print("7th max for column: ", col, " is: ", cs[6])
    if dif_df.loc["score", col] == cs[7]:
        print("8th max for column: ", col, " is: ", cs[7])

print("max(cs) = ", max(cs))
print("min(cs) = ", min(cs))
print("cs: ", cs)
arr = np.array(cs)
percentile95 = np.percentile(arr, 90)
print("90%: ", percentile95)
li = []
for x in cs:
    if x > 0.5:
        li.append(x)
print(len(li))
print(li)

plt.plot(dif_df.loc["score", :])
plt.title("Final scores")
plt.show()

plt.hist(dif_df.loc["score", :], bins=10)
# plt.axvline(x=np.percentile(cs, 90), color='red')
# plt.axvline(x=np.percentile(cs, 90), color='red')
plt.axvline(x=statistics.mean(cs), color='k')
plt.axvline(x=statistics.median(cs), color='y')

plt.axvline(x=statistics.mean(cs) + statistics.stdev(cs), color='red', linestyle='dashed')
plt.axvline(x=statistics.mean(cs) - statistics.stdev(cs), color='red', linestyle='dashed')

plt.axvline(x=statistics.mean(cs) + 2*statistics.stdev(cs), color='green', linestyle='dashed')
plt.axvline(x=statistics.mean(cs) - 2*statistics.stdev(cs), color='green', linestyle='dashed')

plt.axvline(x=statistics.mean(cs) + 3*statistics.stdev(cs), color='orange', linestyle='dashed')
plt.axvline(x=statistics.mean(cs) - 3*statistics.stdev(cs), color='orange', linestyle='dashed')

xmin = statistics.mean(cs) + statistics.stdev(cs)
xmax = statistics.mean(cs) - statistics.stdev(cs)
# plt.axhline(y=50, xmin=xmin, xmax=1, color='red', linestyle='dashed')
# plt.axhline(y=40, xmin=0, xmax=0.1, color='red', linestyle='dashed')
plt.grid(axis='y')
plt.show()

print("STD")
print("stdev:", statistics.stdev(cs))
print("mean:", statistics.mean(cs))
print("median:", statistics.median(cs))


for i in range(0, len(cs)):
    if (abs(cs[i] - statistics.mean(cs))) > statistics.stdev(cs):
        print("hi:", i)

for i in range(0, len(cs)):
    if (abs(cs[i] - statistics.mean(cs))) > 2 * statistics.stdev(cs):
        print("hi2:", i)

# for i in range(4, len(cs)):
#     if statistics.stdev(cs[0:i]) - statistics.stdev(cs[0:i - 1]) * 2 < (
#             statistics.stdev(cs[0:i - 1]) - statistics.stdev(cs[0:i - 2])):
#         print(i)
#
# for i in range(3, len(cs)):
#     print(statistics.stdev(cs[0:i-1]) - statistics.stdev(cs[0:i]))

# for i in range(10, len(cs)):
#     if statistics.stdev(cs[0:i]) > 1.5 * (statistics.stdev(cs[0:i - 1])):
#         print(i)
# print("end2!")
# for i in range(2, len(cs)):
#     print(statistics.stdev(cs[0:i]))


def containers(index1, bool):
    if bool:
        for col in dif_df.columns:
            if dif_df.loc["score", col] == cs[index1]:
                nodes1 = col
                value = cs[index1]
    else:
        nodes1 = index1

    cpu = dm_df.loc['U(CPU)', nodes1]
    mem = dm_df.loc['U(mem)', nodes1]

    if int(nodes1) % 2 == 0:
        max_cpu = cpu / 0.9
        n = int(max_cpu / 200) + 1
        max_cpu = n * 200
        if cpu / max_cpu > 0.9:
            n += 1
    else:
        max_mem = mem / 0.9
        n = int(max_mem / 1000) + 1
        max_mem = n * 1000
        if mem / max_mem > 0.9:
            n += 1

    return n


def containers2(index2, bool):
    if bool:
        for col in dif_df.columns:
            if dif_df.loc["score", col] == cs[index2]:
                nodes2 = col
                value = cs[index2]
    else:
        nodes2 = index2

    cpu = UCPU_pred[int(nodes2) - 1]
    mem = Umem_pred[int(nodes2) - 1]

    if int(nodes2) % 2 == 0:
        max_cpu = cpu / 0.9
        n = int(max_cpu / 200) + 1
        max_cpu = n * 200
        if cpu / max_cpu > 0.9:
            n += 1
        if cpu / max_cpu < 0.1:
            n -= 1
    else:
        mem2 = mem / 0.9
        n = int(mem2 / 1000) + 1
        mem2 = n * 1000
        if mem / mem2 > 0.9:
            n += 1
        if mem / mem2 < 0.1:
            n -= 1
    return n


def containers3(index2):
    for col in dif_df.columns:
        if dif_df.loc["score", col] == cs[index2]:
            nodes2 = col
            value = cs[index2]
    # UCPU_pred[list(DAG2.nodes).index(36)] = 1800
    cpu = UCPU_pred[int(nodes2) - 1]
    mem = Umem_pred[int(nodes2) - 1]
    n = containers(index2, True)

    if int(nodes2) % 2 == 0:
        max_cpu = n * 200
        while (cpu / max_cpu) > 0.9:
            n += 1
            max_cpu = n * 200

        if n > 1:
            while cpu / max_cpu < 0.85 and n > 1:
                if cpu / ((n - 1) * 200) < 0.9:
                    n -= 1
                    max_cpu = n * 200
                else:
                    break

    else:
        max_mem = n * 1000
        while mem / max_mem > 0.9:
            n += 1
            max_mem = n * 1000

        if n > 1:
            while mem / max_mem < 0.85 and n > 1:
                if mem / ((n - 1) * 1000) < 0.9:
                    n -= 1
                    max_mem = n * 1000
                else:
                    break

    return n


print("num of cont: ")
print(containers(0, True))
print(containers2(0, True))
print(containers3(0))

print(containers(1, True))
print(containers3(1))

print("start")
for i in range(0, 11):
    if containers(i, True) > containers2(i, True):
        print('cs ', i, 'scale in ', containers(i, True) - containers2(i, True), 'containers, cs = ', cs[i])
    elif containers(i, True) < containers2(i, True):
        print('cs ', i, 'scale out ', containers2(i, True) - containers(i, True), 'containers, cs = ', cs[i])


# ********** real workload **********
real_edge_workload = pd.DataFrame()

for col in dataframe:
    workloads = []
    for i in range(int(len(dataframe)/10)):
        x = 0
        for j in range(10):
            x += dataframe.loc[i*10+j, col]
        workloads.append(x/10)
    real_edge_workload.loc[:, col] = workloads

graph_list2 = []
for i in range(len(fs_df)):
    graph_list2.append(nx.Graph())
    graph_list2[i] = G.copy()

j = 0
for ind in real_edge_workload.index:
    for edge in graph_list2[j].edges:
        (graph_list2[j])[edge[0]][edge[1]]["weight"] = real_edge_workload[str(edge)][ind]
    j += 1


real_node_workload = pd.DataFrame()
for graph in graph_list2:
    for node in graph.nodes():
        result = 0
        for edge in list(graph.in_edges(node)):
            result += graph[edge[0]][edge[1]]["weight"]
            result = round(result, 4)
        real_node_workload.loc[graph_list2.index(graph), node] = result
# real_node_workload.loc[:, "101"] = API2
print("real node workload len:", len(real_node_workload.columns))
real_node_workload.drop(0, inplace=True, axis=1)

print("real node workload len:", len(real_node_workload.columns))
real_node_workload.to_csv("mini graph/real node workloads.csv")

real_node_workload = pd.read_csv("mini graph/real node workloads.csv", index_col=0)
print("real node workload len:", len(real_node_workload.columns))

scaler1 = MinMaxScaler()
workload_df_scaled = pd.DataFrame(scaler1.fit_transform(workload_df), columns=workload_df.columns)
scaler2 = MinMaxScaler()
real_node_workload_scaled = pd.DataFrame(scaler2.fit_transform(real_node_workload), columns=workload_df.columns)

x1 = workload_df_scaled.iloc[:, 0].to_list()
x2 = real_node_workload_scaled.iloc[:, 0].to_list()

# x1 = workload_df.iloc[-1, :]
# x2 = real_node_workload.iloc[-1, :]
plt.plot(x2[-300:], label='real workload')
plt.plot(x1[-300:], label='calculated workload', )
plt.xticks(np.arange(0, 301, step=20))  # Set label locations.
# plt.yticks(np.arange(1040, 1120, step=5))
plt.legend()
plt.show()


MSE = 0
workload_mse_df = pd.DataFrame()
# for col in workload_df:
for col in range(1, 12):
    ms = sqrt(mean_squared_error(workload_df_scaled.iloc[-300:, workload_df_scaled.columns.get_loc(str(col))],
                                 real_node_workload_scaled.iloc[-300:, real_node_workload_scaled.columns.get_loc(str(col))]))
    workload_mse_df.loc[0, str(col)] = ms
    MSE += ms
MSE /= 100
workload_mse_df.to_csv("mini graph/real_workload_mse.csv")
plt.plot(workload_mse_df.iloc[0, :])
plt.title("MSE for real workload and calculated workload")
plt.xlabel("Node")
plt.ylabel("MSE")
plt.xticks(np.arange(0, 12))
plt.grid(axis='both', linewidth=0.4)
plt.subplots_adjust(left=.15)

# plt.yticks(np.arange(0, 0.0002, step=0.0005))
plt.show()
print("MSE in real workload and calculated is:", MSE)


# **********  RNN ml_prediction **********

# ml36, ml36_rms = ml_prediction("36")
# ml41, ml41_rms = ml_prediction("41")
# ml45, ml45rms = ml_prediction("45")
# ml77, ml77rms = ml_prediction("77")
# ml0, ml0rms = ml_prediction("0")
# ml17, ml17rms = ml_prediction("17")
# ml24, ml24rms = ml_prediction("24")
# ml37, ml37rms = ml_prediction("37")
# ml10, ml10rms = ml_prediction("10")
ml1, ml1rms = ml_prediction("1")

# for x in workload_df:
#     plt.plot(workload_df[x], label=x)
# plt.legend()
# plt.show()
# lin = list(workload_df.loc[int(0.85 * len(workload_df)):, "36"])
# # lin2 = list(workload_df.loc[int(0.7*len(workload_df)):, "36"])
# lin2 = list(workload_df.iloc[-len(ml24):, list(DAG2.nodes).index(24)])
# lin3 = list(workload_df.iloc[-len(ml17):, list(DAG2.nodes).index(17)])
#
# print("len: ", len(ml17))
# print(len(lin3))
# plt.title("node 17")
# plt.scatter(range(len(lin3)), lin3, label="real", color='blue')
# plt.scatter(range(len(ml17)), ml17, label="pred", color='red')
# plt.legend()
# plt.show()
#
# plt.title("node 24")
# plt.scatter(range(len(lin2)), lin2, label="real", color='blue')
# plt.scatter(range(len(ml24)), ml24, label="pred", color='red')
# plt.legend()
# plt.show()
#
# RNN_ = rnn_prediction("1")
# # plt.plot(lin2, label="workload 36")
# # plt.plot(range(21, 21 + len(RNN_)), RNN_, label="RNN")
# # plt.plot(RNN_, label="RNN")
# plt.plot(range(300), RNN_.iloc[-300:, 0], color='red', label='RNN prediction')
# plt.plot(range(300), RNN_.iloc[-300:, 1], color='blue', label='workload')
# # plt.plot(ml45[-len(RNN_):], label="linear reg", color='yellow')
# # plt.plot(ml45[-100:], label="linear reg", color='yellow')
#
# plt.legend(loc='best')
# plt.show()


# number = 0
# rnn_df = pd.DataFrame()
# for node in list(G.nodes):
#     if node == 0:
#         continue
#     number += 1
#     print(number, " wait! we are in node:", node, "!")
#     res = rnn_prediction(str(node))
#     rnn_df[node] = res["test predictions"]
#
# rnn_df.to_csv("mini graph/RNN.csv")


# ************ last process inshaallah!! **********

containers_df = pd.DataFrame(columns=dm_df.columns)
cpu_utilization_df = pd.DataFrame(columns=dif_df.columns)
mem_utilization_df = pd.DataFrame(columns=dif_df.columns)

conts_pred1 = []
for col in dm_df:
    conts_pred1.append(containers(col, False))

# containers_df.loc[len(containers_df)] = conts_pred1
# dif_df = pd.read_csv("mini graph/differenced_dm.csv", index_col=0)
# cpu_utilization_df.loc[len(cpu_utilization_df)] = dif_df.loc['ucpu(t)']
# mem_utilization_df.loc[len(mem_utilization_df)] = dif_df.loc['umem(t)']
print("conts_pred1:", conts_pred1)

time_list = []


def decision_making(timee, n_list):

    start = time.time()
    dm_df2 = pd.DataFrame(columns=fs_df.columns, index=["w(t+1)", "U(CPU)", "U(mem)"])

    # dm_df2.loc["w(t+1)"] = workload_df2.iloc[-2, :]
    dm_df2.loc["w(t+1)"] = predictions2.iloc[-timee+1, :]
    # dm_df2.loc["w(t+1)"] = workload_df.iloc[-timee+1, :]
    dm_df2.loc["U(CPU)"] = UCPU_df.iloc[-timee, :]
    dm_df2.loc["U(mem)"] = Umem_df.iloc[-timee, :]
    UCPU_pred = []
    Umem_pred = []
    for column in dm_df2:
        if int(column) % 2 == 0:
            UCPU_pred.append(round(dm_df2.loc["w(t+1)", column] * 5, 4))  # 4.2
            Umem_pred.append(round(dm_df2.loc["w(t+1)", column] * 0.62, 4))  # 0.01
        else:
            UCPU_pred.append(round(dm_df2.loc["w(t+1)", column] * 1.2, 4))  # 0.01
            Umem_pred.append((round(dm_df2.loc["w(t+1)", column] * 20, 4)))  # 4.88

    dm_df2.loc["cpu pred"] = UCPU_pred
    dm_df2.loc['memory pred'] = Umem_pred
    dm_df2.to_csv("mini graph/predicted2.csv")

    wt = workload_df.iloc[-timee, :]
    # wt = predictions2.iloc[-2, :]
    wt1 = dm_df2.loc['w(t+1)']
    dif_df2 = pd.DataFrame(columns=dm_df.columns)

    conts = []
    conts2 = []
    for i in range(len(n_list)):
        if list(G.nodes)[i+1] % 2 == 0:
            conts.append(n_list[i]*200)
            conts2.append(n_list[i]*25)
        else:
            conts.append(n_list[i]*1000)
            conts2.append(n_list[i]*60)

    divided2 = []
    divided3 = []
    for col in dm_df2:
        if int(col) % 2 == 0:
            n = max(conts[int(col) - 1] / 200, conts2[int(col) - 1] / 25)
            # print("zoj:", conts[list(DAG2.nodes).index(int(col))] / 200, conts2[list(DAG2.nodes).index(int(col))] / 25)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 200) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 25) * 100)

        else:
            n = max(conts[int(col) - 1] / 1000, conts2[int(col) - 1] / 60)
            # print("fard:", conts[list(DAG2.nodes).index(int(col))]/1000, conts2[list(DAG2.nodes).index(int(col))]/60)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 60) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 1000) * 100)

    dif_df2.loc['w(t+1) - w(t)'] = wt1 - wt
    dif_df2.loc['ucpu(t)'] = divided2
    dif_df2.loc['umem(t)'] = divided3

    dif_df3 = pd.DataFrame(columns=dif_df2.columns)
    dif_df3.loc['w(t+1) - w(t)'] = wt1 - wt
    dif_df3.loc['ucpu(t)'] = divided2
    dif_df3.loc['umem(t)'] = divided3

    cpu_utilization_df.loc[len(cpu_utilization_df)] = divided2
    mem_utilization_df.loc[len(mem_utilization_df)] = divided3
    cpu_utilization_df.to_csv("mini graph/cpu_utilization.csv")
    mem_utilization_df.to_csv("mini graph/mem_utilization.csv")

    dif_df2 = dif_df2.round(4)

    dif_df2.to_csv(f'mini graph/dif_dfs/differenced_dm2T{timee}.csv')
    divisors = []
    for j in range(len(dif_df2)):
        row = np.array(dif_df2.iloc[j, :])
        divisors.append(np.sqrt(row @ row))

    for col in dif_df2.columns:
        dif_df2.loc['w(t+1) - w(t)', col] /= divisors[0]
        dif_df2.loc['ucpu(t)', col] /= divisors[1]
        dif_df2.loc['umem(t)', col] /= divisors[2]

    for col in dif_df2.columns:
        weights1 = [0.4, 0.4, 0.2]
        weights2 = [0.4, 0.2, 0.4]
        if int(col) % 2 == 0:
            if dif_df3.loc['ucpu(t)', col] > 90 and dif_df3.loc['umem(t)', col] > 90:
                weights1 = [0.2, 1, 1]
            elif dif_df3.loc['ucpu(t)', col] > 90:
                weights1 = [0.2, 0.8, 0.2]
            elif dif_df3.loc['umem(t)', col] > 90:
                weights1 = [0.2, 0.4, 0.4]
            dif_df2.loc['w(t+1) - w(t)', col] *= weights1[0]
            dif_df2.loc['ucpu(t)', col] *= weights1[1]
            dif_df2.loc['umem(t)', col] *= weights1[2]
        else:
            if dif_df3.loc['umem(t)', col] > 90 and dif_df3.loc['ucpu(t)', col] > 90:
                weights2 = [0.2, 1, 1]
            elif dif_df3.loc['umem(t)', col] > 90:
                weights2 = [0.2, 0.2, 0.8]
            elif dif_df3.loc['ucpu(t)', col] > 90:
                weights2 = [0.2, 0.4, 0.4]

            dif_df2.loc['w(t+1) - w(t)', col] *= weights2[0]
            dif_df2.loc['ucpu(t)', col] *= weights2[1]
            dif_df2.loc['umem(t)', col] *= weights2[2]

    a_pos = []
    a_neg = []
    for j in range(len(dif_df2)):
        row = dif_df2.iloc[j, :]
        max_val = np.max(row)
        min_val = np.min(row)

        a_pos.append(max_val)
        a_neg.append(min_val)

    sp = []
    sn = []
    cs = []
    for col in dif_df2.columns:
        diff_pos = pow((dif_df2.loc['w(t+1) - w(t)', col] - a_pos[0]), 2)
        diff_pos += pow(dif_df2.loc['ucpu(t)', col] - a_pos[1], 2)
        diff_pos += pow(dif_df2.loc['umem(t)', col] - a_pos[2], 2)
        diff_pos = sqrt(diff_pos)

        diff_neg = pow(dif_df2.loc['w(t+1) - w(t)', col] - a_neg[0], 2)
        diff_neg += pow(dif_df2.loc['ucpu(t)', col] - a_neg[1], 2)
        diff_neg += pow(dif_df2.loc['umem(t)', col] - a_neg[2], 2)
        diff_neg = sqrt(diff_neg)

        sp.append(diff_pos)
        sn.append(diff_neg)
        cs.append(sn[-1] / (sp[-1] + sn[-1]))

    dif_df2.loc["pos_dist"] = sp
    dif_df2.loc["neg_dist"] = sn
    dif_df2.loc["score"] = cs

    dif_df.to_csv("mini graph/normalized2.csv")

    if timee == 2 or timee == 3:
        print("time: ", timee)

        cs.sort(reverse=True)

        scale_out = []
        scale_in = []

        for i in range(0, len(cs)):
            if (cs[i] - statistics.mean(cs)) > statistics.stdev(cs):
                scale_out.append(cs[i])
        for i in range(0, len(cs)):
            if (statistics.mean(cs) - cs[i]) > 2 * statistics.stdev(cs):
                scale_in.append(cs[i])

        for sc in scale_out:
            for col in dif_df2.columns:
                if dif_df2.loc["score", col] == sc:
                    print("scale out for column: ", col, " score: ", sc)

        for sc in scale_in:
            for col in dif_df2.columns:
                if dif_df2.loc["score", col] == sc:
                    print("scale in for column: ", col, " score: ", sc)

    conts_pred = []
    for col in dm_df2:
        cpu = UCPU_pred[int(col) - 1]
        mem = Umem_pred[int(col) - 1]
        if (dif_df2.loc['score', col] - statistics.mean(cs) > 1 * statistics.stdev(cs)) or \
                (statistics.mean(cs) - dif_df2.loc['score', col] > 2 * statistics.stdev(cs)):
            print("Time", timee, "bye: ", col)

            if int(col) % 2 == 0:
                max_cpu = cpu / 0.9
                n1 = int(max_cpu / 200) + 1
                max_cpu = n1 * 200
                if cpu / max_cpu > 0.9:
                    n1 += 1
                if cpu / max_cpu < 0.1:
                    n1 -= 1
                max_mem = mem/0.9
                n2 = int(max_mem/25) + 1
                n = max(n1, n2)
            else:
                mem2 = mem / 0.9
                n1 = int(mem2 / 1000) + 1
                mem2 = n1 * 1000
                if mem / mem2 > 0.9:
                    n1 += 1
                if mem / mem2 < 0.1:
                    n1 -= 1
                cpu2 = cpu/0.9
                n2 = int(cpu2/60) + 1
                n = max(n1, n2)
            conts_pred.append(n)

        else:
            conts_pred.append(n_list[int(col) - 1])

        end = time.time()
        time_list.append((end - start))
        # print("elapsed time:", end - start)
    containers_df.loc[len(containers_df)] = conts_pred
    containers_df.to_csv("mini graph/containers.csv")
    return conts_pred



# n3 = decision_making(4, conts_pred1)
# n2 = decision_making(3, n3)
# n1 = decision_making(2, n2)

ns = []
ns.append(decision_making(372, conts_pred1))
for i in range(371, 1, -1):
    ns.append(decision_making(i, ns[-1]))

print("avg elapsed time:", statistics.mean(time_list))

memory_util = pd.read_csv("mini graph/mem_utilization.csv", index_col=0)
memory_util_mean = memory_util.mean(axis=1)
plt.plot(memory_util_mean)
plt.title('Average memory utilization of microservices')
plt.xlabel('Time')
plt.ylabel('Memory utilization')
plt.grid(axis='both', linewidth=0.4)
plt.show()

plt.hist(memory_util)
plt.xticks(np.arange(70, 101, step=10))
plt.title("memory utilization histogram")
plt.show()

cpu_util = pd.read_csv("mini graph/cpu_utilization.csv", index_col=0)
cpu_util_mean = cpu_util.mean(axis=1)
plt.plot(cpu_util_mean)
plt.title('Average cpu utilization of microservices')
plt.xlabel('Time')
plt.ylabel('CPU utilization')
plt.yticks(np.arange(86.8, 89.4, step=0.2))
plt.grid(axis='both', linewidth=0.4)

plt.show()

plt.hist(cpu_util)
plt.xticks(np.arange(70, 101, step=10))
plt.title("CPU utilization histogram")
plt.show()

# containers_df = pd.read_csv("mini graph/containers.csv", index_col=0)
# containers_df = pd.read_csv("mini graph/containers.csv", index_col=0)
# conts45 = containers_df["45"].tolist()
# del conts45[0]
# plt.plot(conts45, label='proactive')
# plt.plot(cont3_45, label='reactive')
# plt.yticks(np.arange(min(min(conts45), min(cont3_45)), max(max(conts45), max(cont3_45))+1, step=1))
# plt.xlabel("time")
# plt.ylabel("resources")
# plt.legend()
# plt.show()
#
# num_react = 0
# for i in range(len(cont3_45)-1):
#     if cont3_45[i] != cont3_45[i+1]:
#         num_react += 1
#
# num_proact = 0
# for i in range(len(conts45)-1):
#     if conts45[i] != conts45[i+1]:
#         num_proact += 1
#
# print("number of reactive scaling:", num_react)
# print("number of proactive scaling:", num_proact)
#
# plt.plot(range(300), workload_df.iloc[-300:, list(G.nodes).index(0)], label='workload', color='blue')
# plt.plot(range(300), predictions2.iloc[-300:, list(G.nodes).index(0)], label='RNN prediction', color='red')
# plt.legend()
# plt.show()

cpu_rem_df = []
for row in range(len(cpu_util)):
    rem = 0
    num = 0
    for col in cpu_util.columns:
        if cpu_util.loc[row, col] > 90:
            rem += cpu_util.loc[row, col] - 90
    rem /= 100
    cpu_rem_df.append(rem)

plt.plot(cpu_rem_df)
plt.title("CPU SLA violation percentile")
plt.xlabel("time")
plt.ylabel("SLA violation")
plt.yticks(np.arange(0, 0.016, step=0.0025))
plt.grid(axis='both', linewidth=0.4)
plt.show()

memory_rem_df = []
for row in range(len(memory_util)):
    rem = 0
    num = 0
    for col in memory_util.columns:
        if memory_util.loc[row, col] > 90:
            rem += memory_util.loc[row, col] - 90
    rem /= 100
    memory_rem_df.append(rem)

plt.plot(memory_rem_df)
plt.title("memory SLA violation percentile")
plt.xlabel("time")
plt.ylabel("SLA violation")
plt.yticks(np.arange(0, 0.007, step=0.001))
plt.grid(axis='both', linewidth=0.4)
plt.show()
