# @title { vertical-output: true}

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

# ********** modules **********

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

    # pr_df.to_csv("big graph/pr_df.csv")
    knn = KNeighborsRegressor(n_neighbors=2)
    lin_model = LinearRegression()
    lasso_model = Lasso(alpha=10)
    ridge_model = Ridge(alpha=10)
    en_model = ElasticNet(alpha=0.5, l1_ratio=0.4, max_iter=10000000)
    tree_model = DecisionTreeRegressor()
    rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
    lassocv_model = LassoCV(cv=5)
    poly_model = PolynomialFeatures(degree=2, include_bias=False)
    poly_reg_model = LinearRegression()
    xgboot_model = GradientBoostingRegressor(max_depth=4, learning_rate=0.1)
    # parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1.5, 10], 'gamma': [1e-7, 1e-4],
    #               'epsilon': [0.1, 0.2, 0.5, 0.3]}
    SVR_model = SVR(C=1500000, epsilon=0.1)
    from sklearn.model_selection import GridSearchCV
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
    en_model.fit(xtrain, ytrain)
    tree_model.fit(xtrain, ytrain)
    rf_model.fit(xtrain, ytrain.ravel())
    lassocv_model.fit(xtrain, ytrain)
    poly_model.fit(poly_model.fit_transform(xtrain), ytrain)
    poly_reg_model.fit(poly_model.fit_transform(xtrain), ytrain)
    xgboot_model.fit(xtrain, ytrain)
    SVR_model.fit(xtrain, ytrain)
    # clf.fit(xtrain, ytrain)
    # print("best params for SVR:", clf.best_params_)

    print("Training set score: {:.2f}".format(lasso_model.score(xtrain, ytrain)))
    print("Test set score: {:.2f}".format(lasso_model.score(xtest, ytest)))

    knn_pred = knn.predict(xtest)
    lr_pred = lin_model.predict(xtest)
    lasso_pred = lasso_model.predict(xtest)
    ridge_pred = ridge_model.predict(xtest)
    elastic_pred = en_model.predict(xtest)
    tree_pred = tree_model.predict(xtest)
    rf_pred = rf_model.predict(xtest)
    lassocv_pred = lassocv_model.predict(xtest)
    xgboot_pred = xgboot_model.predict(xtest)
    SVR_pred = SVR_model.predict(xtest)

    knn_rmse = sqrt(mean_squared_error(ytest[-347:, ], knn_pred[-347:, ]))
    print("KNN ML MSE: ", knn_rmse, "for node: ", node)

    lr_rmse = sqrt(mean_squared_error(ytest[-347:, ], lr_pred[-347:, ]))
    print("LR ML MSE: ", lr_rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], lasso_pred[-347:, ]))
    print("Lasso ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], ridge_pred[-347:, ]))
    print("Ridge ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], tree_pred[-347:, ]))
    print("tree ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], elastic_pred[-347:, ]))
    print("EN ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], rf_pred[-347:, ]))
    print("RF ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], xgboot_pred[-347:, ]))
    print("XGboot ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], SVR_pred[-347:, ]))
    print("SVR ML MSE: ", rmse, "for node: ", node)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], lassocv_pred[-347:, ]))
    print("Lasso cv ML MSE: ", rmse)

    rmse = sqrt(mean_squared_error(ytest[-347:, ], poly_reg_model.predict(poly_model.fit_transform(xtest))[-347:, ]))
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
    dataset = data.values

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))

    print("dataset shape", dataset.shape)
    train_size = int(len(dataset) * 0.8)

    train, val, test = dataset[:train_size, :], dataset[train_size:, :], dataset[train_size:, :]
    window_size = 9
    data2_x, data2_y = df_to_x_y(dataset, window_size)
    x2_train, x2_test, y2_train, y2_test = train_test_split(data2_x, data2_y, train_size=0.8, shuffle=False,
                                                            random_state=42)

    x_train, y_train = df_to_x_y(train, window_size)
    x_val, y_val = df_to_x_y(val, window_size)
    x_test, y_test = df_to_x_y(test, window_size)

    start2 = time.time()
    model1 = Sequential()
    model1.add(InputLayer((9, 1)))
    model1.add(LSTM(100, batch_size=64))
    model1.add(Dense(1, 'relu'))
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
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
    #                               patience=50, min_lr=0)
    hist = model1.fit(x_train, y_train, validation_split=0.2, epochs=1200,
                      callbacks=[save_best, stopping, lr_scheduler])
    end2 = time.time()
    print("avg train time:", end2-start2)

    train_predictions = model1.predict(x_train).flatten()
    train_predictions = scaler.inverse_transform(train_predictions.reshape(-1, 1))
    y_train = scaler.inverse_transform(y_train.reshape(-1, 1))

    train_results = pd.DataFrame(data={'train predictions': list(train_predictions), 'actual': list(y_train)})
    train_results.to_csv("big graph/train.csv")
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

    start1 = time.time()
    test_predictions = model1.predict(x_test).flatten()
    end1 = time.time()
    print("rnn prediction time:", end1 - start1)
    test_predictions = scaler.inverse_transform(test_predictions.reshape(-1, 1))
    # x_test = scaler.inverse_transform(x_test.reshape(-1, 1))
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    test_results = pd.DataFrame(data={'test predictions': list(test_predictions), 'actual': list(y_test)})
    test_results.to_csv("big graph/test.csv")

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
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean absolute error')
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
    plt.ylabel('Mean squared error')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    evaluation = model1.evaluate(x_test, y_test)
    for i in range(len(model1.metrics_names)):
        print("metric:", model1.metrics_names[i], " : ", evaluation[i])

    return test_results


def scheduler(epoch, lr):
    if epoch < 50:
        return 0.01
    if epoch < 350:
        return 0.001
    else:
        return 0.0001


# ********** dataset **********

# G = nx.gnp_random_graph(100, 0.07, directed=False)
# print(nx.is_connected(G))
#
# pos = nx.spring_layout(G, seed=5)
# nx.draw_networkx_nodes(
#     G, pos, linewidths=1, node_size=200, node_color='pink', alpha=1,
# )
# nx.draw_networkx_edges(G, pos, width=1)
# nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
#
# # nx.draw_networkx_edge_labels(
# #     G, pos,
# #     edge_labels={edge: edge for edge in G.edges()},
# #     font_color='red', font_size=5
# # )
# ax = plt.gca()
# ax.margins(0)
# plt.axis("tight")
# plt.tight_layout()
# plt.show()
#
#
#
# DAG = nx.DiGraph([(u, v, {'weight': random.randint(-10, 10)}) for (u, v) in G.edges() if u < v])
# DAG.add_node(101)
# for i in range(50):
#     DAG.add_edge(101, i)
#
# print(nx.is_directed_acyclic_graph(DAG))
# # print(networkx.info(DAG))
# print(DAG.number_of_edges())
# print(DAG.number_of_nodes())
#
# pos = nx.spring_layout(DAG, seed=5)
# nx.draw_networkx_nodes(
#     DAG, pos, linewidths=1, node_size=200, node_color='pink', alpha=1,
# )
#
# nx.draw_networkx_edges(DAG, pos, width=1)
# nx.draw_networkx_labels(DAG, pos, font_size=10, font_family="sans-serif")
#
# ax = plt.gca()
# ax.margins(0)
# plt.axis("tight")
# plt.tight_layout()
# plt.title("DAG")
# plt.show()
#
# # save graph object to file
# pickle.dump(DAG, open('big_graph.pickle', 'wb'))

# load graph object from file
DAG2 = pickle.load(open('big_graph.pickle', 'rb'))

pos = nx.spring_layout(DAG2, seed=5)
nx.draw_networkx_nodes(
    DAG2, pos, linewidths=1, node_size=200, node_color='pink', alpha=1,
)

nx.draw_networkx_edges(DAG2, pos, width=1)
nx.draw_networkx_labels(DAG2, pos, font_size=10, font_family="sans-serif")
ax = plt.gca()
ax.margins(0)
plt.axis("tight")
plt.tight_layout()
plt.title("dag2")
plt.show()
nx.write_gexf(DAG2, "test.gexf")

in_deg_centrality = in_degree_centrality(DAG2)
in_deg_centrality[101] = 1


def in_centrality(node):
    if node == 101:
        result = 1
    else:
        result = 0
        for path in nx.all_simple_paths(DAG2, source=101, target=node):
            res = 1
            for i in range(len(path)):
                res *= in_deg_centrality[path[i]]

            result += res
    return result

from networkx import degree_centrality, in_degree_centrality, closeness_centrality

# centrality = degree_centrality(DAG2)
# print("centralities of G2: ", centrality)
# # in_deg_centrality = in_degree_centrality(DAG2)
# print("in degree centrality: ", in_deg_centrality)
# betweenness = nx.betweenness_centrality(DAG2, k=10, normalized=True, endpoints=True)
# print("betweenness of G nodes: ", betweenness)
#
# close_centrality = closeness_centrality(DAG2)
# centrality_list = []
# for node in DAG2:
#     centrality_list.append(close_centrality[node])
#     centrality_list.append(close_centrality[node])
# centrality_list.sort()
# print(centrality_list)
# centrality_list = list(dict.fromkeys(centrality_list))
# for node in DAG2:
#     if close_centrality[node] == centrality_list[-1]:
#         print("1 maximum degree centrality in G is ", centrality_list[-1], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-2]:
#         print("2 maximum degree centrality in G is ", centrality_list[-2], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-3]:
#         print("3 maximum degree centrality in G is ", centrality_list[-3], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-4]:
#         print("4 maximum degree centrality in G is ", centrality_list[-4], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-5]:
#         print("5 maximum degree centrality in G is ", centrality_list[-5], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-6]:
#         print("6 maximum degree centrality in G is ", centrality_list[-6], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-7]:
#         print("7 maximum degree centrality in G is ", centrality_list[-7], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-8]:
#         print("8 maximum degree centrality in G is ", centrality_list[-8], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-9]:
#         print("9 maximum degree centrality in G is ", centrality_list[-9], "\nand it belongs to ", node)
#     elif close_centrality[node] == centrality_list[-10]:
#         print("10 maximum degree centrality in G is ", centrality_list[-10], "\nand it belongs to ", node)

# **********  first step: generate data for edges  **********
# edge_list = []
# indx = 0
# dataframe = pd.DataFrame(columns=list(DAG2.edges))
#
# for edge in DAG2.edges:
#     print("edge 0: ", edge[0])
#     if edge[0] == 101:
#         data = [random.choice([520, 530, 550, 560, 580, 600])]
#     elif edge[0] in [0, 1, 2, 3, 4, 5]:
#         data = [random.choice([151, 152, 153, 154])]
#     elif edge[0] in [5, 6, 7, 8, 9, 10]:
#         data = [random.choice([142, 143, 144, 145])]
#     elif edge[0] in [10, 11, 12, 13, 14, 15]:
#         data = [random.choice([133, 134, 136])]
#     elif edge[0] in [15, 16, 17, 18, 19, 20]:
#         data = [random.choice([125, 126, 127])]
#     elif edge[0] in [20, 21, 22, 23, 24, 25]:
#         data = [random.choice([115, 116, 118])]
#     elif edge[0] in [25, 26, 27, 28, 29, 30]:
#         data = [random.choice([105, 106, 108])]
#     elif edge[0] in [30, 31, 32, 33, 34, 35]:
#         data = [random.choice([96, 97, 98])]
#     elif edge[0] in [35, 36, 37, 38, 39, 40]:
#         data = [random.choice([88, 89, 90, 92])]
#     elif edge[0] in [40, 41, 42, 43, 44, 45]:
#         data = [random.choice([81, 82, 83])]
#     elif edge[0] in [45, 46, 47, 48, 49, 50]:
#         data = [random.choice([75, 76, 77])]
#     elif edge[0] in [50, 51, 52, 53, 54, 55]:
#         data = [random.choice([68, 69, 70])]
#     elif edge[0] in [55, 56, 57, 58, 59, 60]:
#         data = [random.choice([60, 61, 62, 63])]
#     elif edge[0] in [60, 61, 62, 63, 64, 65]:
#         data = [random.choice([53, 54, 55, 56])]
#     elif edge[0] in [65, 66, 67, 68, 69, 70]:
#         data = [random.choice([46, 47, 48])]
#     elif edge[0] in [70, 71, 72, 73, 74, 75]:
#         data = [random.choice([39, 40, 41])]
#     elif edge[0] in [75, 76, 77, 78, 79, 80]:
#         data = [random.choice([32, 33, 34, 35])]
#     elif edge[0] in [80, 81, 82, 83, 84, 85,  86, 87, 88, 89, 90]:
#         data = [random.choice([26, 27, 28, 29])]
#     elif edge[0] in [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]:
#         data = [random.choice([18, 19, 20, 21])]
#
#
#     edge_list.append(data)
#     indx += 1
#
# print("edge list length: ", len(edge_list))
#
# edges45 = []
# nodes45 = []
# for path in nx.all_simple_edge_paths(DAG2, 101, 45):
#     print(path)
#     for x in path:
#         edges45.append(x)
#         for y in x:
#             nodes45.append(y)
# print("nodes 45:", nodes45)
# nodes45 = list(dict.fromkeys(nodes45))
# mylist = list(dict.fromkeys(edges45))
# print("edges45: ", edges45)
# DAG45 = nx.DiGraph()
# DAG45.add_edges_from(edges45)
# nx.write_gexf(DAG45, "graph45.gexf")
#
# edges36 = []
# for path in nx.all_simple_edge_paths(DAG2, 101, 36):
#     print(path)
#     for x in path:
#         edges36.append(x)
# mylist = list(dict.fromkeys(edges36))
# print("edges36: ", edges36)
#
# edges36.remove((101, 2))
# edges36.remove((101, 3))
# edges36.remove((3, 8))
# print("edges36: ", edges36)
#
# timeout = time.time() + 60 * 15 # 15 minutes from now
# i = 0
# while time.time() < timeout:
#     for li in edge_list:
#         index = edge_list.index(li)
#         if list(DAG2.edges)[index] in edges45:
#             if time.time() < timeout - 870:
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.3, 0.5])
#             elif (time.time() > timeout - 870) and (time.time() < timeout - 600):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 600) and (time.time() < timeout - 580):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 580) and (time.time() < timeout - 400):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 400) and (time.time() < timeout - 380):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 380) and (time.time() < timeout - 350):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 350) and (time.time() < timeout - 330):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 330) and (time.time() < timeout - 200):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif (time.time() > timeout - 200) and (time.time() < timeout - 180):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 180) and (time.time() < timeout - 5):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
#             elif time.time() > timeout - 5:
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.2, 0.4, 0.4])
#
#         elif list(DAG2.edges)[index] in edges36:
#             if time.time() < timeout - 850:
#                 x = li[i] + numpy.random.choice([-2, -1, 1, 2], p=[0.15, 0.25, 0.35, 0.25])
#             elif (time.time() > timeout - 850) and (time.time() < timeout-830):
#                 x = li[i] + numpy.random.choice([1, 0, -1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 830) and (time.time() < timeout-600):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.29, 0.4, 0.31])
#             elif (time.time() > timeout - 600) and (time.time() < timeout - 580):
#                 x = li[i] + numpy.random.choice([1, 0, -1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 580) and (time.time() < timeout - 500):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.29, 0.4, 0.31])
#             elif (time.time() > timeout - 500) and (time.time() < timeout - 480):
#                 x = li[i] + numpy.random.choice([1, 0, -1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 450) and (time.time() < timeout-400):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.29, 0.4, 0.31])
#             elif (time.time() > timeout - 400) and (time.time() < timeout - 380):
#                 x = li[i] + numpy.random.choice([1, 0, -1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 380) and (time.time() < timeout-200):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.29, 0.4, 0.31])
#             elif (time.time() > timeout - 200) and (time.time() < timeout - 180):
#                 x = li[i] + numpy.random.choice([1, 0, -1], p=[0.2, 0.4, 0.4])
#             elif (time.time() > timeout - 180) and (time.time() < timeout-5):
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.29, 0.4, 0.31])
#             elif time.time() > timeout - 5:
#                 x = li[i] + numpy.random.choice([1, 0, -1], p=[0.2, 0.4, 0.4])
#
#         else:
#             x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.25, 0.4, 0.35])
#             if time.time() > timeout - 850:
#                 x = li[i] + numpy.random.choice([-1, 0, 1], p=[0.15, 0.7, 0.15])
#         li.append(x)
#         edge_list[index] = li
#     i += 1
#     # time.sleep(0.01)
#
#
# for (li, edge) in zip(edge_list, DAG2.edges):
#     dataframe[edge] = li
#
#
# dataframe.to_csv("big graph/big graph edges.csv")
# print("done!!!!!!")
dataframe = pd.read_csv("big graph/big graph edges.csv", index_col=0)
dataframe.plot()
plt.show()

API = []
for row in range(len(dataframe)):
    x = 0
    for column in range(1, 51):
        x += dataframe.iloc[row, -column]
    API.append(x)

# ********** second step: build graph with weights **********

# graph_list = []
# for i in range(len(dataframe)):
#     graph_list.append(nx.Graph())
#     graph_list[i] = DAG2.copy()
#
# j = 0
# for ind in dataframe.index:
#     for edge in graph_list[j].edges:
#         (graph_list[j])[edge[0]][edge[1]]["weight"] = dataframe[str(edge)][ind]
#     j += 1

# pos = nx.spring_layout(graph_list[0], seed=5)
# nx.draw_networkx_nodes(
#     graph_list[0], pos, linewidths=1, node_size=200, node_color='pink', alpha=1,
# )
#
# nx.draw_networkx_edges(graph_list[0], pos, width=1)
# nx.draw_networkx_labels(graph_list[0], pos, font_size=10, font_family="sans-serif")
# edge_labels = nx.get_edge_attributes(graph_list[0], "weight")
# nx.draw_networkx_edge_labels(graph_list[0], pos, edge_labels)
# ax = plt.gca()
# ax.margins(0)
# plt.axis("tight")
# plt.tight_layout()
# plt.title("graph list 0")
# plt.show()


# ********** second step: convert probabilistic edge data **********

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
#         if node == 101:
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
# edge_df = pd.DataFrame.from_dict(lists)
# edge_df.to_csv('big graph/p_edge.csv')
# print("done1")

edge_df = pd.read_csv('big graph/p_edge.csv', index_col=0)
graph_list2 = []
for i in range(len(edge_df)):
    graph_list2.append(nx.Graph())
    graph_list2[i] = DAG2.copy()

j = 0
for ind in edge_df.index:
    for edge in graph_list2[j].edges:
        (graph_list2[j])[edge[0]][edge[1]]["weight"] = edge_df[str(edge)][ind]
    j += 1

nx.write_gexf(graph_list2[-1], "graph_list2.gexf")
pickle.dump(graph_list2[-1], open('big graph/graph_list2.pickle', 'wb'))
betweenness = nx.betweenness_centrality(graph_list2[-1], normalized=True, weight="weight")
c = nx.load_centrality(graph_list2[-1], weight="weight", normalized=True)
d = nx.pagerank(graph_list2[-2], weight="weight")
close = nx.closeness_centrality(graph_list2[-1], distance="weight")
in_degree = nx.in_degree_centrality(graph_list2[-1])

print("betweenness of last nodes: ", sorted(betweenness.items(), key=lambda x:x[1], reverse=True))
print("closeness of last nodes: ", sorted(close.items(), key=lambda x:x[1]))
print("in deg centrality of last nodes: ", sorted(in_degree.items(), key=lambda x:x[1], reverse=True))
print("load centrality of last nodes: ", sorted(c.items(), key=lambda x:x[1], reverse=True))
print("page rank of last nodes: ", sorted(d.items(), key=lambda x:x[1], reverse=True))

from networkx.algorithms.community.centrality import girvan_newman
from random import randint
from networkx.algorithms.community import greedy_modularity_communities
import networkx.algorithms.community as nx_comm

G2 = graph_list2[-2]
modu_communities = list(greedy_modularity_communities(G2))

cc = []
for tt in modu_communities:
    cc.append(len(tt))

print("\ngreedy modularity found ", len(modu_communities), "communities")
print("modularity: ", modu_communities)
# color_map = []
# colors = []
# n = len(modu_communities)
# for i in range(n):
#     colors.append('#%06X' % randint(0, 0xFFFFFF))
# colors2 = ['#CFFF67', '#F37449', '#CFB9A4', '#9FFCFD', '#F5B8FD', '#A2ECFD']
# for node in G2:
#     for ng in range(len(modu_communities)):
#         if node in modu_communities[ng]:
#             color_map.append(colors2[ng])
# plt.title("greedy modularity")
# pos = nx.spring_layout(G2, k=0.55, iterations=20)
# nx.draw(G2, node_color=color_map, with_labels=True, edgelist=[], node_size=200, font_size=8, alpha=0.8,
#         pos=pos)
# plt.show()
# xx = []
# for x in modu_communities:
#     xx.append(len(x))
# xx = sorted(xx, reverse=True)
# b = 0
# for xxx in range(len(xx)):
#     if xx[xxx] < 11 and b == 0:
#         b = xxx
# print("greedy modularity communities bigger than 10: ", xx[0:b])
# print("greedy modularity's modularity: ", nx_comm.modularity(G2, modu_communities))

# ********** third step: calculate probabilities for each node **********

# node_df = pd.DataFrame(columns=list(DAG2.nodes))
#
# lists1 = []
# number = len(edge_df)
# for graph in graph_list:
#     lists = []
#     for node in graph.nodes():
#         result = 0
#         for path in nx.all_simple_paths(graph, source=101, target=node):
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
# node_df.to_csv("big graph/p_node.csv")
# print("done2!")
node_df = pd.read_csv("big graph/p_node.csv", index_col=0)

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
#         phfs_df.loc[j - 1, str(col)] = [collection]
#         for key in collection:
#             f_score += float(key) * collection[key]
#         fs_df.loc[j - 1, str(col) + ""] = float(format(f_score, '.4f'))
#
# phfs_df.to_csv("big graph/phfs.csv")
# fs_df.to_csv("big graph/function_scores.csv")

phfs_df = pd.read_csv("big graph/phfs.csv", index_col=0)
fs_df = pd.read_csv("big graph/function_scores.csv", index_col=0)
fs_df.drop("101", inplace=True, axis=1)

API2 = []
for i in range(int(len(API) / 10)):
    x = 0
    for j in range(10):
        x += API[i * 10 + j]
    API2.append(x / 10)

# print("API2: ", API2)
# ********** prediction **********

workload_df = fs_df.copy()

workload_df.loc[:, "API"] = API2
workload_df.loc[:, workload_df.columns != "API"] = \
    round(workload_df.loc[:, workload_df.columns != "API"].multiply(workload_df["API"], axis="index"), 4)
workload_df.drop("API", inplace=True, axis=1)

workload_df.to_csv("big graph/workloads.csv")

# workload_df = pd.read_csv("big graph/workloads.csv", index_col=0)
# plt.tight_layout(pad=0.1, h_pad=-0.7, w_pad=0.2)
# plt.show()

i = 1
for col in range(75, 100):
    # if col>24 and col < 50:
    plt.subplot(5, 5, i)
    plt.title(col, fontsize=8, pad=1, color='red')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=5)
    plt.plot(workload_df[str(col)])
    i += 1
plt.tight_layout(pad=0.1)
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

UCPU_df.to_csv("big graph/CPU util.csv")
Umem_df.to_csv("big graph/memory util.csv")


# ********** decision making table **********

dm_df = pd.DataFrame(columns=fs_df.columns, index=["w(t+1)", "U(CPU)", "U(mem)"])

predictions2 = pd.read_csv("big graph/RNN.csv", index_col=0)
# dm_df.loc["w(t+1)"] = workload_df2.iloc[-2, :]
dm_df.loc["w(t+1)"] = predictions2.iloc[-347, :]
# dm_df.loc["w(t+1)"] = workload_df.iloc[-345, :]
dm_df.loc["U(CPU)"] = UCPU_df.iloc[-348, :]
dm_df.loc["U(mem)"] = Umem_df.iloc[-348, :]

dm_df.to_csv("big graph/predicted.csv")

UCPU_pred = []
Umem_pred = []
for column in dm_df:
    if int(column) % 2 == 0:
        UCPU_pred.append(round(dm_df.loc["w(t+1)", column] * 5, 4))  # 4.2
        Umem_pred.append(round(dm_df.loc["w(t+1)", column] * 0.62, 4))  # 0.01
    else:
        UCPU_pred.append(round(dm_df.loc["w(t+1)", column] * 1.2, 4))  # 0.01
        Umem_pred.append((round(dm_df.loc["w(t+1)", column] * 20, 4)))  # 4.88

print("U CPU(t+1): ", UCPU_pred)
print("U memory(t+1): ", Umem_pred)


# ********* resources **********

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
wt = workload_df.iloc[-348, :]
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
        n = max(conts[list(DAG2.nodes).index(int(col))]/200, conts2[list(DAG2.nodes).index(int(col))]/25)
        divided2.append(dm_df.loc['U(CPU)', col] / (n*200) * 100)
        divided3.append(dm_df.loc['U(mem)', col] / (n*25) * 100)

    else:
        n = max(conts[list(DAG2.nodes).index(int(col))]/1000, conts2[list(DAG2.nodes).index(int(col))]/60)
        divided2.append(dm_df.loc['U(CPU)', col] / (n*60) * 100)
        divided3.append(dm_df.loc['U(mem)', col] / (n*1000) * 100)

ucpu = dm_df.loc['U(CPU)']
umem = dm_df.loc['U(mem)']


dif_df.loc['w(t+1) - w(t)'] = wt1 - wt
dif_df.loc['ucpu(t)'] = divided2
dif_df.loc['umem(t)'] = divided3

dif_df = dif_df.round(4)
dif_df.to_csv("big graph/differenced_dm.csv")

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

dif_df.to_csv("big graph/normalized.csv")

for row in dif_df.index:
    print(f'Max element of row {row} is:', max(dif_df.loc[row]), ' and its belong to ', )

cs.sort(reverse=True)
for col in dif_df.columns:
    if dif_df.loc["score", col] == cs[0]:
        print("0th max for column: ", col, " is: ", cs[0])
    if dif_df.loc["score", col] == cs[1]:
        print("1th max for column: ", col, " is: ", cs[1])
    if dif_df.loc["score", col] == cs[2]:
        print("2th max for column: ", col, " is: ", cs[2])
    if dif_df.loc["score", col] == cs[3]:
        print("3th max for column: ", col, " is: ", cs[3])
    if dif_df.loc["score", col] == cs[4]:
        print("4th max for column: ", col, " is: ", cs[4])
    if dif_df.loc["score", col] == cs[5]:
        print("5th max for column: ", col, " is: ", cs[5])
    if dif_df.loc["score", col] == cs[6]:
        print("6th max for column: ", col, " is: ", cs[6])
    if dif_df.loc["score", col] == cs[7]:
        print("7th max for column: ", col, " is: ", cs[7])
    if dif_df.loc["score", col] == cs[8]:
        print("8th max for column: ", col, " is: ", cs[8])
    if dif_df.loc["score", col] == cs[9]:
        print("9th max for column: ", col, " is: ", cs[9])
    if dif_df.loc["score", col] == cs[10]:
        print("10th max for column: ", col, " is: ", cs[10])
    if dif_df.loc["score", col] == cs[94]:
        print("94th max for column: ", col, " is: ", cs[94])
    if dif_df.loc["score", col] == cs[95]:
        print("95th max for column: ", col, " is: ", cs[95])
    if dif_df.loc["score", col] == cs[96]:
        print("96th max for column: ", col, " is: ", cs[96])
    if dif_df.loc["score", col] == cs[97]:
        print("97th max for column: ", col, " is: ", cs[97])
    if dif_df.loc["score", col] == cs[98]:
        print("98th max for column: ", col, " is: ", cs[98])
    if dif_df.loc["score", col] == cs[99]:
        print("99th max for column: ", col, " is: ", cs[99])

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
plt.scatter(95, np.percentile(arr, 95), color='red')
plt.axvline(x=95, color='red')
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
plt.grid(axis='both', linewidth=0.4)
plt.show()

plt.boxplot(dif_df.loc["score", :], whis=1)
plt.show()
print("STD")
print("stdev:", statistics.stdev(cs))
print("mean:", statistics.mean(cs))
print("median:", statistics.median(cs))

z_score = stats.zscore(cs)
for i in range(0, len(cs)):
    if z_score[i] > 1 or z_score[i]<-1:
        print("hi0:", i)

for i in range(0, len(cs)):
    if (abs(cs[i] - statistics.mean(cs))) > statistics.stdev(cs):
        print("hi:", i)

for i in range(0, len(cs)):
    if (abs(cs[i] - statistics.mean(cs))) > 2 * statistics.stdev(cs):
        print("hi2:", i)


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

    cpu = UCPU_pred[list(DAG2.nodes).index(int(nodes2))]
    mem = Umem_pred[list(DAG2.nodes).index(int(nodes2))]

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
    cpu = UCPU_pred[list(DAG2.nodes).index(int(nodes2))]
    mem = Umem_pred[list(DAG2.nodes).index(int(nodes2))]
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
for i in range(100):
    if containers(i, True) > containers2(i, True):
        print('cs ', i, 'scale in ', containers(i, True) - containers2(i, True), 'containers, cs = ', cs[i])
    elif containers(i, True) < containers2(i, True):
        print('cs ', i, 'scale out ', containers2(i, True) - containers(i, True), 'containers, cs = ', cs[i])


# ********** real workload **********

# real_edge_workload = pd.DataFrame()
#
# for col in dataframe:
#     workloads = []
#     for i in range(int(len(dataframe)/10)):
#         x = 0
#         for j in range(10):
#             x += dataframe.loc[i*10+j, col]
#         workloads.append(x/10)
#     real_edge_workload.loc[:, col] = workloads
#
# graph_list2 = []
# for i in range(len(fs_df)):
#     graph_list2.append(nx.Graph())
#     graph_list2[i] = DAG2.copy()
#
# j = 0
# for ind in real_edge_workload.index:
#     for edge in graph_list2[j].edges:
#         (graph_list2[j])[edge[0]][edge[1]]["weight"] = real_edge_workload[str(edge)][ind]
#     j += 1
#
#
# real_node_workload = pd.DataFrame()
# for graph in graph_list2:
#     for node in graph.nodes():
#         result = 0
#         for edge in list(graph.in_edges(node)):
#             result += graph[edge[0]][edge[1]]["weight"]
#             result = round(result, 4)
#         real_node_workload.loc[graph_list2.index(graph), node] = result
# # real_node_workload.loc[:, "101"] = API2
# print("real node workload len:", len(real_node_workload.columns))
# real_node_workload.drop(101, inplace=True, axis=1)
#
# print("real node workload len:", len(real_node_workload.columns))
# real_node_workload.to_csv("big graph/real node workloads.csv")

real_node_workload = pd.read_csv("big graph/real node workloads.csv", index_col=0)
print("real node workload len:", len(real_node_workload.columns))

scaler1 = MinMaxScaler()
workload_df_scaled = pd.DataFrame(scaler1.fit_transform(workload_df), columns=workload_df.columns)
scaler2 = MinMaxScaler()
real_node_workload_scaled = pd.DataFrame(scaler2.fit_transform(real_node_workload), columns=workload_df.columns)

x1 = workload_df.iloc[:, list(DAG2.nodes).index(0)].to_list()
x2 = real_node_workload.iloc[:, list(DAG2.nodes).index(0)].to_list()

# x1 = workload_df.iloc[-1, :]
# x2 = real_node_workload.iloc[-1, :]
plt.plot(x2[-300:], label='Real workload')
plt.plot(x1[-300:], label='Calculated workload', )
plt.xlabel("Time")
plt.ylabel("Workload")
plt.xticks(np.arange(0, 301, step=20))  # Set label locations.
plt.yticks(np.arange(1045, 1120, step=5))
plt.legend()
plt.show()

MSE = 0
workload_mse_df = pd.DataFrame()
workload_acc_df = pd.DataFrame()

# for col in workload_df:
for col in range(0, 100):
    ms = sqrt(mean_squared_error(workload_df_scaled.iloc[-300:, workload_df_scaled.columns.get_loc(str(col))],
                                 real_node_workload_scaled.iloc[-300:, real_node_workload_scaled.columns.get_loc(str(col))]))
    workload_mse_df.loc[0, str(col)] = ms

    accs_per_col = []
    for index in range(1, 301):
        print("col:", col)
        error = abs(workload_df_scaled.iloc[-index, workload_df_scaled.columns.get_loc(str(col))] -
                    real_node_workload_scaled.iloc[-index, real_node_workload_scaled.columns.get_loc(str(col))]
                    )/real_node_workload_scaled.iloc[-index, real_node_workload_scaled.columns.get_loc(str(col))]
        acc = (1-error)*100
        accs_per_col.append(acc)
    workload_acc_df[str(col)] = accs_per_col

    MSE += ms
MSE /= 100
workload_mse_df.to_csv("big graph/real_workload_mse.csv")
workload_acc_df.to_csv("big graph/real_workload_acc.csv")

plt.plot(workload_mse_df.iloc[0, :])
plt.title("MSE for real workload and calculated workload")
plt.xlabel("Node")
plt.ylabel("MSE")
plt.xticks(np.arange(0, 101, step=5))
plt.yticks(np.arange(0, 0.014, step=0.002))
plt.grid(axis='both', linewidth=0.4)
plt.show()
print("MSE in real workload and calculated is:", MSE)
# workload_df = scaler1.inverse_transform(workload_df)
# real_node_workload = scaler2.inverse_transform(real_node_workload)

workload_acc_df['mean'] = workload_acc_df.mean(axis=1)
accuracy = workload_acc_df['mean'].mean()
print("accuracy in workload estimation is: ", accuracy)
# **********  RNN ml_prediction **********

ml2, ml2rms = ml_prediction("2")
ml_prediction("3")
ml_prediction("4")

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
# RNN_ = rnn_prediction("0")
# # plt.plot(lin2, label="workload 36")
# # plt.plot(range(21, 21 + len(RNN_)), RNN_, label="RNN")
# # plt.plot(RNN_, label="RNN")
# plt.plot(range(300), RNN_.iloc[-300:, 0], color='red', label='RNN prediction')
# plt.plot(range(300), RNN_.iloc[-300:, 1], color='blue', label='Workload')
# plt.xlabel("Time")
# plt.ylabel("Workload")
# plt.legend()
# plt.plot(ml45[-len(RNN_):], label="linear reg", color='yellow')
# plt.plot(ml45[-100:], label="linear reg", color='yellow')

# plt.legend(loc='best')
# plt.show()


# number = 0
# rnn_df = pd.DataFrame()
# for node in list(DAG2.nodes):
#     if list(DAG2.nodes).index(node) > 69 and list(DAG2.nodes).index(node) < 100 :
#         number += 1
#         print(number, " wait! we are in node:", node, "!")
#         res = rnn_prediction(str(node))
#         rnn_df[node] = res["test predictions"]
#
# rnn_df.to_csv("big graph/rnn3.csv")
# ********** ***********

react_mem = pd.DataFrame()
def containers4(index):
    cpu = UCPU_df.iloc[-1 * index - 1, list(DAG2.nodes).index(45)]
    mem = Umem_df.iloc[-1 * index - 1, list(DAG2.nodes).index(45)]
    max_mem = mem / 0.9
    n1 = int(max_mem / 1000) + 1
    max_mem = n1 * 1000
    if mem / max_mem > 0.9:
        n1 += 1
    max_cpu = cpu / 0.9
    n2 = int(max_cpu/60) + 1
    max_mem = n2 * 60
    if cpu /max_cpu > 0.9:
        n2 += 1
    n = max(n1, n2)
    return n


def containers5(index2):
    cpu = UCPU_pred45[index2]
    mem = Umem_pred45[index2]
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
    return n


pred45 = predictions2["45"]
UCPU_pred45 = []
Umem_pred45 = []

for x in pred45:
    UCPU_pred45.append(round(x * 1.2, 4))  # 0.01
    Umem_pred45.append((round(x * 20, 4)))  # 4.16

cont1_45 = []
for ind in range(len(UCPU_pred45)):
    cont1_45.append(containers4(ind))
cont1_45.reverse()

cont2_45 = []
for ind in range(len(UCPU_pred45)):
    cont2_45.append(containers5(ind))

cont3_45 = cont1_45.copy()

del cont1_45[0]
del cont2_45[0]
# del cont1_45[1]
# del cont2_45[1]
# del cont1_45[2]
# del cont2_45[2]
del cont3_45[-1]
# del cont3_45[-2]
# del cont3_45[-3]

print("main containers for node 45: ", cont1_45)
print("predicted containers for node 45: ", cont2_45)
print("reactive containers for node 45:", cont3_45)

# plt.plot(cont1_45, label='main')
# plt.plot(cont2_45, label='pred')
# plt.plot(cont3_45, label='reactive')
# plt.legend()
# plt.show()


# ************ just for comparison **********

containers_df = pd.DataFrame(columns=dm_df.columns)
containers_df2 = pd.DataFrame(columns=dm_df.columns)
containers_df3 = pd.DataFrame(columns=dm_df.columns)

cpu_utilization_df = pd.DataFrame(columns=dif_df.columns)
mem_utilization_df = pd.DataFrame(columns=dif_df.columns)
cpu_utilization_df2 = pd.DataFrame(columns=dif_df.columns)
mem_utilization_df2 = pd.DataFrame(columns=dif_df.columns)

conts_pred1 = []
for col in dm_df:
    conts_pred1.append(max(containers(col, False), containers2(col, False)))

containers_df.loc[len(containers_df)] = conts_pred1
containers_df2.loc[len(containers_df2)] = conts_pred1
containers_df3.loc[len(containers_df3)] = conts_pred1

dif_df = pd.read_csv("big graph/differenced_dm.csv", index_col=0)
# cpu_utilization_df.loc[len(cpu_utilization_df)] = dif_df.loc['ucpu(t)']
# mem_utilization_df.loc[len(mem_utilization_df)] = dif_df.loc['umem(t)']
# cpu_utilization_df2.loc[len(cpu_utilization_df2)] = dif_df.loc['ucpu(t)']
# mem_utilization_df2.loc[len(mem_utilization_df2)] = dif_df.loc['umem(t)']
print("conts_pred1:", conts_pred1)

time_list = []


def decision_making(timee, n_list):

    start = time.time()
    dm_df2 = pd.DataFrame(columns=fs_df.columns, index=["w(t+1)", "U(CPU)", "U(mem)"])

    dm_df2.loc["w(t+1)"] = predictions2.iloc[-timee+1, :]
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
    dm_df2.to_csv("big graph/predicted2.csv")

    wt = workload_df.iloc[-timee, :]
    wt1 = dm_df2.loc['w(t+1)']
    dif_df2 = pd.DataFrame(columns=dm_df.columns)

    conts = []
    conts2 = []
    for i in range(len(n_list)):
        if list(DAG2.nodes)[i] % 2 == 0:
            conts.append(n_list[i]*200)
            conts2.append(n_list[i]*25)
        else:
            conts.append(n_list[i]*1000)
            conts2.append(n_list[i]*60)

    divided2 = []
    divided3 = []
    for col in dm_df2:
        if int(col) % 2 == 0:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 200, conts2[list(DAG2.nodes).index(int(col))] / 25)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 200) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 25) * 100)

        else:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 1000, conts2[list(DAG2.nodes).index(int(col))] / 60)
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
    cpu_utilization_df.to_csv("big graph/cpu_utilization.csv")
    mem_utilization_df.to_csv("big graph/mem_utilization.csv")

    dif_df2 = dif_df2.round(4)

    dif_df2.to_csv(f'big graph/dif_dfs/differenced_dm2T{timee}.csv')
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

    dif_df.to_csv("big graph/normalized2.csv")

    if timee == 2 or timee==3:
        print("time: ", timee)

        cs.sort(reverse=True)

        scale_out = []
        scale_in = []

        for i in range(0, len(cs)):
            if (cs[i] - statistics.mean(cs)) > statistics.stdev(cs):
                scale_out.append(cs[i])
        for i in range(0, len(cs)):
            if (statistics.mean(cs) - cs[i]) > 3 * statistics.stdev(cs):
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
        cpu = UCPU_pred[list(DAG2.nodes).index(int(col))]
        mem = Umem_pred[list(DAG2.nodes).index(int(col))]
        if (dif_df2.loc['score', col] - statistics.mean(cs) > 1 * statistics.stdev(cs)) or \
                (statistics.mean(cs) - dif_df2.loc['score', col] > 3 * statistics.stdev(cs)):
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
            conts_pred.append(n_list[list(DAG2.nodes).index(int(col))])

        end = time.time()
        time_list.append((end - start))
        # print("elapsed time:", end - start)
    containers_df.loc[len(containers_df)] = conts_pred
    containers_df.to_csv("big graph/containers.csv")
    return conts_pred


def decision_making2(timee, n_list):
    dm_df2 = pd.DataFrame(columns=fs_df.columns, index=["w(t+1)", "U(CPU)", "U(mem)"])
    dm_df2.loc["w(t+1)"] = predictions2.iloc[-timee + 1, :]
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

    conts = []
    conts2 = []
    for i in range(len(n_list)):
        if list(DAG2.nodes)[i] % 2 == 0:
            conts.append(n_list[i]*200)
            conts2.append(n_list[i]*25)
        else:
            conts.append(n_list[i]*1000)
            conts2.append(n_list[i]*60)

    divided2 = []
    divided3 = []
    for col in dm_df2:
        if int(col) % 2 == 0:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 200, conts2[list(DAG2.nodes).index(int(col))] / 25)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 200) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 25) * 100)

        else:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 1000, conts2[list(DAG2.nodes).index(int(col))] / 60)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 60) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 1000) * 100)

    cpu_utilization_df2.loc[len(cpu_utilization_df2)] = divided2
    mem_utilization_df2.loc[len(mem_utilization_df2)] = divided3
    cpu_utilization_df2.to_csv("big graph/cpu_utilization2.csv")
    mem_utilization_df2.to_csv("big graph/mem_utilization2.csv")

    conts_pred = []
    for col in dm_df2:
        if int(col) % 2 == 0:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 200, conts2[list(DAG2.nodes).index(int(col))] / 25)
            cpu = UCPU_pred[list(DAG2.nodes).index(int(col))] / (n * 200) * 100
            mem = Umem_pred[list(DAG2.nodes).index(int(col))] / (n * 25) * 100
        else:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 1000, conts2[list(DAG2.nodes).index(int(col))] / 60)
            cpu = UCPU_pred[list(DAG2.nodes).index(int(col))] / (n * 60) * 100
            mem = Umem_pred[list(DAG2.nodes).index(int(col))] / (n * 1000) * 100
        last_cpu = dm_df2.loc["U(CPU)"] = UCPU_df.iloc[-timee, UCPU_df.columns.get_loc(col)]
        last_mem = dm_df2.loc["U(mem)"] = Umem_df.iloc[-timee, Umem_df.columns.get_loc(col)]

        if(cpu>90 or mem>90 or cpu<70 or mem<70):
            n = containers_df2.iloc[-1, containers_df2.columns.get_loc(col)]
            if (cpu > 90 or mem > 90):
                n += 1
            elif (cpu < 70 or mem < 70):
                n -= 1
            conts_pred.append(n)
        else:
            conts_pred.append(n_list[list(DAG2.nodes).index(int(col))])
    if timee == 348:
        print("time 348: UCPU_pred", UCPU_pred)
        print("time 348 Umem_pred:", Umem_pred)

    containers_df2.loc[len(containers_df2)] = conts_pred
    containers_df2.to_csv("big graph/containers2.csv")
    return conts_pred


def decision_making3(timee, n_list):
    dm_df2 = pd.DataFrame(columns=fs_df.columns, index=["w(t+1)", "U(CPU)", "U(mem)"])
    dm_df2.loc["w(t+1)"] = predictions2.iloc[-timee + 1, :]
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

    conts = []
    conts2 = []
    for i in range(len(n_list)):
        if list(DAG2.nodes)[i] % 2 == 0:
            conts.append(n_list[i]*200)
            conts2.append(n_list[i]*25)
        else:
            conts.append(n_list[i]*1000)
            conts2.append(n_list[i]*60)

    divided2 = []
    divided3 = []
    for col in dm_df2:
        if int(col) % 2 == 0:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 200, conts2[list(DAG2.nodes).index(int(col))] / 25)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 200) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 25) * 100)

        else:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 1000, conts2[list(DAG2.nodes).index(int(col))] / 60)
            divided2.append(dm_df2.loc['U(CPU)', col] / (n * 60) * 100)
            divided3.append(dm_df2.loc['U(mem)', col] / (n * 1000) * 100)

    cpu_utilization_df2.loc[len(cpu_utilization_df2)] = divided2
    mem_utilization_df2.loc[len(mem_utilization_df2)] = divided3
    cpu_utilization_df2.to_csv("big graph/cpu_utilization3.csv")
    mem_utilization_df2.to_csv("big graph/mem_utilization3.csv")

    conts_pred = []
    for col in dm_df2:
        cpu = UCPU_pred[list(DAG2.nodes).index(int(col))]
        mem = Umem_pred[list(DAG2.nodes).index(int(col))]
        if int(col) % 2 == 0:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 200, conts2[list(DAG2.nodes).index(int(col))] / 25)
            cpu2 = UCPU_pred[list(DAG2.nodes).index(int(col))] / (n * 200) * 100
            mem2 = Umem_pred[list(DAG2.nodes).index(int(col))] / (n * 25) * 100
        else:
            n = max(conts[list(DAG2.nodes).index(int(col))] / 1000, conts2[list(DAG2.nodes).index(int(col))] / 60)
            cpu2 = UCPU_pred[list(DAG2.nodes).index(int(col))] / (n * 60) * 100
            mem2 = Umem_pred[list(DAG2.nodes).index(int(col))] / (n * 1000) * 100

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
                if mem / mem2 < 0.7:
                    n1 -= 1
                cpu2 = cpu/0.9
                n2 = int(cpu2/60) + 1
                n = max(n1, n2)

        conts_pred.append(n)

    containers_df3.loc[len(containers_df3)] = conts_pred
    containers_df3.to_csv("big graph/containers3.csv")

    return conts_pred


ns = []
ns.append(decision_making(348, conts_pred1))
for i in range(347, 1, -1):
    ns.append(decision_making(i, ns[-1]))

ns2 = []
ns2.append(decision_making2(348, conts_pred1))
for i in range(347, 1, -1):
    ns2.append(decision_making2(i, ns2[-1]))

ns3 = []
ns3.append(decision_making3(348, conts_pred1))
for i in range(347, 1, -1):
    ns3.append(decision_making3(i, ns3[-1]))

print("avg elapsed time:", statistics.mean(time_list))

memory_util = pd.read_csv("big graph/mem_utilization.csv", index_col=0)
memory_util_mean = memory_util.mean(axis=1)
plt.plot(memory_util_mean)
plt.title('average memory utilization of microservices')
plt.xlabel('Time')
plt.ylabel('Memory utilization')
plt.grid(axis='both', linewidth=0.4)
plt.show()

plt.hist(memory_util)
plt.xticks(np.arange(70, 101, step=10))
plt.title("Memory utilization histogram")
plt.show()

memory_util2 = pd.read_csv("big graph/mem_utilization2.csv", index_col=0)
memory_util_mean2 = memory_util2.mean(axis=1)
plt.plot(memory_util_mean2)
plt.title('Average memory utilization of microservices2')
plt.xlabel('Time')
plt.ylabel('Memory utilization')
plt.show()


cpu_util = pd.read_csv("big graph/cpu_utilization.csv", index_col=0)
cpu_util_mean = cpu_util.mean(axis=1)
plt.plot(cpu_util_mean)
plt.title('average cpu utilization of microservices')
plt.xlabel('Time')
plt.ylabel('CPU utilization')
plt.yticks(np.arange(86.2, 88.4, step=0.2))
plt.grid(axis='both', linewidth=0.4)
plt.show()

cpu_util2 = pd.read_csv("big graph/cpu_utilization2.csv", index_col=0)
cpu_util_mean2 = cpu_util2.mean(axis=1)
plt.plot(cpu_util_mean2)
plt.title('Average cpu utilization of microservices2')
plt.xlabel('Time')
plt.ylabel('CPU utilization')
# plt.yticks(np.arange(86.2, 88.4, step=0.2))
plt.show()

plt.hist(cpu_util)
plt.xticks(np.arange(70, 101, step=10))
plt.title("CPU utilization histogram")
plt.show()


no_usage_cpu = pd.read_csv("big graph/cpu_utilization.csv", index_col=0)
for row in range(len(no_usage_cpu)):
    for col in range(len(no_usage_cpu.columns)):
        if no_usage_cpu.iloc[row, col] < 90:
            no_usage_cpu.iloc[row, col] = 90 - no_usage_cpu.iloc[row, col]
        else:
            no_usage_cpu.iloc[row, col] = 0
no_usage_cpu['mean'] = no_usage_cpu.mean(axis=1)
no_usage_cpu.to_csv("big graph/ no_CPU.csv")

no_usage_cpu2 = pd.read_csv("big graph/cpu_utilization2.csv", index_col=0)
for row in range(len(no_usage_cpu2)):
    for col in range(len(no_usage_cpu2.columns)):
        if no_usage_cpu2.iloc[row, col] < 90:
            no_usage_cpu2.iloc[row, col] = 90 - no_usage_cpu2.iloc[row, col]
        else:
            no_usage_cpu2.iloc[row, col] = 0
no_usage_cpu2['mean'] = no_usage_cpu2.mean(axis=1)
no_usage_cpu2.to_csv("big graph/no_CPU2.csv")


plt.plot(no_usage_cpu['mean'], label='Proposed approach')
plt.plot(no_usage_cpu2['mean'], label='Hybrid approach without conscious selection of microservices')
plt.xlabel('Time')
plt.ylabel('Average of unused CPU percentile')
plt.legend()
plt.grid(axis='both', linewidth=0.4)
plt.show()
x = (no_usage_cpu['mean'].mean() - no_usage_cpu2['mean'])/no_usage_cpu2['mean'].mean()
print("improvement of unused cpu: ", x)

no_usage_mem = pd.read_csv("big graph/mem_utilization.csv", index_col=0)
for row in range(len(no_usage_mem)):
    for col in range(len(no_usage_mem.columns)):
        if no_usage_mem.iloc[row, col] < 90:
            no_usage_mem.iloc[row, col] = 90 - no_usage_mem.iloc[row, col]
        else:
            no_usage_mem.iloc[row, col] = 0
no_usage_mem['mean'] = no_usage_mem.mean(axis=1)
no_usage_mem.to_csv("big graph/no_mem.csv")

no_usage_mem2 = pd.read_csv("big graph/mem_utilization2.csv", index_col=0)
for row in range(len(no_usage_mem2)):
    for col in range(len(no_usage_mem2.columns)):
        if no_usage_mem2.iloc[row, col] < 90:
            no_usage_mem2.iloc[row, col] = 90 - no_usage_mem2.iloc[row, col]
        else:
            no_usage_cpu2.iloc[row, col] = 0
no_usage_mem2['mean'] = no_usage_mem2.mean(axis=1)
no_usage_mem2.to_csv("big graph/no_mem2.csv")


plt.plot(no_usage_mem['mean'], label='Proposed approach')
plt.plot(no_usage_mem2['mean'], label='Hybrid approach without conscious selection of microservices')
plt.xlabel('Time')
plt.ylabel('Average of unused memory percentile')
plt.legend()
plt.grid(axis='both', linewidth=0.4)
plt.show()

improve = (no_usage_mem['mean'].mean() - no_usage_mem2['mean'])/no_usage_mem2['mean'].mean()
print("improvement of unused memory: ", improve)

containers_df = pd.read_csv("big graph/containers.csv", index_col=0)
conts45 = containers_df["45"].tolist()
del conts45[0]
plt.plot(conts45, label='proactive')
plt.plot(cont3_45, label='reactive')
plt.yticks(np.arange(min(min(conts45), min(cont3_45)), max(max(conts45), max(cont3_45))+1, step=1))
plt.xlabel("time")
plt.ylabel("resources")
plt.legend()
plt.grid(axis='both', linewidth=0.4)
plt.show()


scale_num1 = []
for i in range(len(containers_df) - 1):
    res = 0
    for col in containers_df.columns:
        if containers_df.iloc[i, containers_df.columns.get_loc(str(col))] != containers_df.iloc[i + 1, containers_df.columns.get_loc(str(col))]:
            res += 1
    scale_num1.append(res)

scale_num2 = []
for i in range(len(containers_df2) - 1):
    res = 0
    for col in containers_df2.columns:
        if containers_df2.iloc[i, containers_df2.columns.get_loc(str(col))] != containers_df2.iloc[i + 1, containers_df2.columns.get_loc(str(col))]:
            res += 1
    scale_num2.append(res)

scale_num3 = []
for i in range(len(containers_df3) - 1):
    res = 0
    for col in containers_df3.columns:
        if containers_df3.iloc[i, containers_df3.columns.get_loc(str(col))] != containers_df3.iloc[i + 1, containers_df3.columns.get_loc(str(col))]:
            res += 1
    scale_num3.append(res)

plt.plot(scale_num1, label='Proposed approach')
# plt.plot(scale_num2, label='hybrid')
plt.plot(scale_num3, label='Proactive approach without conscious selection of microservices')
plt.xlabel('Time')
plt.ylabel('Number of scales')
plt.legend()
plt.grid(axis='both', linewidth=0.4)
plt.show()
print("improvement in number of scales: ", (statistics.mean(scale_num1) - statistics.mean(scale_num3))/statistics.mean(scale_num3))

containers_df['sum'] = containers_df.sum(axis=1)
containers_df2['sum'] = containers_df2.sum(axis=1)
containers_df3['sum'] = containers_df3.sum(axis=1)

plt.plot(containers_df['sum'], label="Proposed approach")
plt.plot(containers_df2['sum'], label='Hybrid approach without conscious selection of microservices')
# plt.plot(containers_df3['sum'], label='proactive')
print("improvement in sum of microservices: ", (containers_df['sum'].mean()-containers_df2['sum'].mean())/containers_df2['sum'].mean())


plt.legend()
plt.xlabel("Time")
plt.ylabel("Sum of microservices resources")
plt.grid(axis='both', linewidth=0.4)
plt.show()

num_react = 0
for i in range(len(cont3_45)-1):
    if cont3_45[i] != cont3_45[i+1]:
        num_react += 1

num_proact = 0
for i in range(len(conts45)-1):
    if conts45[i] != conts45[i+1]:
        num_proact += 1

print("number of reactive scaling:", num_react)
print("number of proactive scaling:", num_proact)

plt.plot(range(300), workload_df.iloc[-300:, list(DAG2.nodes).index(0)], label='Workload', color='blue')
plt.plot(range(300), predictions2.iloc[-300:, list(DAG2.nodes).index(0)], label='RNN prediction', color='red')
plt.xlabel("Time")
plt.ylabel("Workload")
plt.legend()
plt.show()

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
plt.xlabel("Time")
plt.ylabel("SLA violation")
plt.yticks(np.arange(0, 0.1, step=0.01))
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
plt.title("Memory SLA violation percentile")
plt.xlabel("Time")
plt.ylabel("SLA violation")
plt.yticks(np.arange(0, 0.0275, step=0.0025))
plt.grid(axis='both', linewidth=0.4)
plt.show()
