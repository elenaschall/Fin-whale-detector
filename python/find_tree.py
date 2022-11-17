import datetime
import pathlib
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm

from detect_f20p import *


tk1_range = [2.5, 5.5]
tk2_range = [40, 70]
tsf_range = [2, 10]
tst_range = [-8, 5]
tbw_range = [75, 95]


def plot_roc_curve(estimator, y, x, name):
    # Plot ROC curves in the test set
    y_pred = estimator.predict_proba(x)[:, 1]
    fpr, tpr, thresholds_list = metrics.roc_curve(y, y_pred)
    fpr_1percent = np.argmin(np.abs(fpr - 0.01))
    tpr_1percernt = tpr[fpr_1percent]
    threshold = thresholds_list[fpr_1percent]
    print('TPR for 1% FPR: ', tpr_1percernt)
    print('Treshold where that happens: ', threshold)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=estimator)
    display.plot()
    plt.title(name)
    plt.show()


def plot_roc_curve_from_probab(y_prob, y, name):
    # Plot ROC curves in the test set
    fpr, tpr, thresholds_list = metrics.roc_curve(y, y_prob)
    fpr_1percent = np.argmin(np.abs(fpr - 0.01))
    tpr_1percernt = tpr[fpr_1percent]
    threshold = thresholds_list[fpr_1percent]
    print('TPR for 1% FPR: ', tpr_1percernt)
    print('Treshold where that happens: ', threshold)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    display.plot()
    plt.title(name)
    plt.show()


def find_best_threshold(th_name, column_var, operator, node_parent, th_range, node_to_check='both'):
    fin_prop = node_parent.F20P.sum() / len(node_parent)
    impurity = fin_prop * (1 - fin_prop) * 2
    threshold_list = np.linspace(th_range[0], th_range[1], 100)
    impurities_list = []
    best_th = threshold_list[0]
    best_left = node_parent
    best_right = []
    for th in threshold_list:
        if operator == '<':
            decision_mask = node_parent[column_var] < th
        elif operator == '>':
            decision_mask = node_parent[column_var] > th
        else:
            raise Exception('This is not a valid operator')
        left_node = node_parent.loc[decision_mask]
        right_node = node_parent.loc[~decision_mask]

        prop_left_node = left_node.F20P.sum() / len(left_node)
        prop_right_node = right_node.F20P.sum() / len(right_node)

        if node_to_check == 'both':
            impurity_left_node = prop_left_node * (1 - prop_left_node) * 2
            impurity_right_node = prop_right_node * (1 - prop_right_node) * 2
            impurity_decision = (impurity_left_node * len(left_node) +
                                 impurity_right_node * len(right_node)) / len(decision_mask)
            probab = impurity_left_node
        elif node_to_check == 'right':
            impurity_right_node = prop_right_node * (1 - (1 - prop_right_node)) * 2
            impurity_decision = (impurity_right_node * len(right_node)) / len(decision_mask)
            probab = impurity_right_node
        elif node_to_check == 'left':
            impurity_left_node = prop_left_node * (1 - prop_left_node) * 2
            impurity_decision = (impurity_left_node * len(left_node)) / len(decision_mask)
            probab = impurity_left_node

        impurities_list.append(impurity_decision)
        if impurity_decision < impurity:
            impurity = impurity_decision
            best_th = th
            best_left = left_node
            best_right = right_node
            best_probab = probab

    print('best threshold ', th_name, best_th)

    plt.plot(threshold_list, impurities_list)
    plt.xlabel('%s values' % th_name)
    plt.ylabel('impurity')
    plt.show()

    return impurity, best_left, best_right, best_th, best_probab


def find_best_thresholds_first_split(th_name1, th_name2, column_var1, column_var2, operator, node_parent, th1_range, th2_range):
    fin_prop = node_parent.F20P.sum() / len(node_parent)
    impurity = fin_prop * (1 - fin_prop) * 2
    threshold_list1 = np.linspace(th1_range[0], th1_range[1], 100)
    threshold_list2 = np.linspace(th2_range[0], th2_range[1], 100)
    impurities_list = []
    best_th1 = threshold_list1[0]
    best_th2 = threshold_list2[0]
    best_left = node_parent
    best_right = []
    for th1 in threshold_list1:
        for th2 in threshold_list2:
            if operator == '<':
                decision_mask = (node_parent[column_var1] < th1) & (node_parent[column_var2] < th2)
            elif operator == '>':
                decision_mask = (node_parent[column_var1] > th1) & (node_parent[column_var2] > th2)
            else:
                raise Exception('This is not a valid operator')

            left_node = node_parent.loc[decision_mask]
            right_node = node_parent.loc[~decision_mask]

            prop_left_node = left_node.F20P.sum() / len(left_node)
            prop_right_node = right_node.F20P.sum() / len(right_node)

            impurity_left_node = prop_left_node * (1 - prop_left_node) * 2
            impurity_right_node = prop_right_node * (1 - prop_right_node) * 2

            # impurity_decision = (impurity_left_node * len(left_node) +
            #                      impurity_right_node * len(right_node)) / len(decision_mask)
            impurity_decision = (impurity_right_node * len(right_node)) / len(decision_mask)

            impurities_list.append(impurity_decision)
            if impurity_decision < impurity:
                impurity = impurity_decision
                best_th1 = th1
                best_th2 = th2
                best_left = left_node
                best_right = right_node
                probab = impurity_right_node

        print('best thresholds ', th_name1, best_th1, th_name2, best_th2)

        plt.plot(impurities_list)
        plt.xlabel('%s values' % th_name1)
        plt.xlabel('%s values' % th_name2)
        plt.ylabel('impurity')
        plt.show()

        return impurity, best_left, best_right, best_th1, best_th2, probab


# -----------------------------------------------------------------------------
# Load the data
N_TIMES_BALANCED = 3
fin_whale_path = pathlib.Path('./test/data/TkurtAll_Miller2.csv')
all_fin_detections = pd.read_csv(fin_whale_path, parse_dates=['DateTime'])
all_fin_detections['BW'] = all_fin_detections['BW'] / 23 * 100
n_fin_detections = all_fin_detections.F20P.sum()

# Get the paper thresholds
TK1 = 2.5
TK2 = 40
TSF = 8
TST = -2
TST2 = -8
TK1_2 = 4.75
TBW = 75

print('predicting old model...')
selected_rows = apply_thresholds(all_fin_detections, TK1, TK2, TSF, TST, TST2, TBW, TK1_2)
all_fin_detections['prediction'] = all_fin_detections.index.isin(selected_rows.index).astype(int)
selected_rows = filter_lonely_detections(selected_rows)
all_fin_detections['corrected_prediction'] = all_fin_detections.index.isin(selected_rows.index).astype(int)

y_test = all_fin_detections['F20P']
all_fin_detections.to_csv('./test/data/TkurtAll_Miller_thresholds_paper_still_wrong.csv')

# Score the model!
print('With the paper thresholds')
cm = confusion_matrix(y_test, all_fin_detections.corrected_prediction)
print('-------------------------------------------------')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, all_fin_detections.corrected_prediction)}')
print(f'Area Under Curve: {roc_auc_score(y_test, all_fin_detections.corrected_prediction)}')
print(f'Recall or TPR score: {recall_score(y_test, all_fin_detections.corrected_prediction)}')
print(f'Precision score: {precision_score(y_test, all_fin_detections.corrected_prediction)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')

print('With the paper thresholds no filtering')
cm = confusion_matrix(y_test, all_fin_detections.prediction)
print('-------------------------------------------------')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, all_fin_detections.prediction)}')
print(f'Area Under Curve: {roc_auc_score(y_test, all_fin_detections.prediction)}')
print(f'Recall or TPR score: {recall_score(y_test, all_fin_detections.prediction)}')
print(f'Precision score: {precision_score(y_test, all_fin_detections.prediction)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')

# THE LEFT SPLIT IS THE YES, RIGHT SPLIT IS THE NO
impurity12, tk1_or_tk2_left, tk1_or_tk2_right, TK1, TK2, probab0 = find_best_thresholds_first_split('TK1', 'TK2',
                                                                                                    'Kurt',
                                                                                                    'KurtProd', '>',
                                                                                                    all_fin_detections,
                                                                                                    tk1_range,
                                                                                                    tk2_range)
# TSF split
impurity3, tsf_left, tsf_right, TSF, probab1 = find_best_threshold('TSF', 'SNRF', '>', tk1_or_tk2_left,
                                                                   tsf_range, 'right')

# TST split
impurity4, tst_left, tst_right, TST, probab2 = find_best_threshold('TST', 'SNRT', '>', tsf_left, tst_range, 'right')

# TST.2 split
impurity5, tst_left2, tst_right2, TST2, probab3 = find_best_threshold('TST.2', 'SNRT', '>', tst_left, tst_range, 'left')

# TBW split
impurity6, tbw_left, tbw_right, TBW, probab4 = find_best_threshold('TBW', 'BW', '>', tst_right2, tbw_range, 'right')

# TK1.2 split
impurity6, tbw_left, tbw_right, TK1_2, probab5 = find_best_threshold('TK1.2', 'Kurt', '>', tbw_left, tk1_range, 'both')

selected_rows, df_with_prob = apply_thresholds_probab(all_fin_detections, TK1, TK2, TSF, TST, TST2, TBW, TK1_2,
                                        probab0, probab1, probab2, probab3, probab4, probab5)
plot_roc_curve_from_probab(df_with_prob['probab'], df_with_prob['F20P'], 'manual DT with probab')

all_fin_detections['prediction'] = all_fin_detections.index.isin(selected_rows.index).astype(int)

selected_rows = filter_lonely_detections(selected_rows)
all_fin_detections['corrected_prediction'] = all_fin_detections.index.isin(selected_rows.index).astype(int)
y_test = all_fin_detections['F20P']

# Score the model!
print('With the found thresholds')
print('-------------------------------------------------')
cm = confusion_matrix(y_test, all_fin_detections.corrected_prediction)
print(f'Confusion Matrix: \n{cm}')
print(f'Area Under Curve: {roc_auc_score(y_test, all_fin_detections.corrected_prediction)}')
print(f'Recall or TPR score: {recall_score(y_test, all_fin_detections.corrected_prediction)}')
print(f'Precision score: {precision_score(y_test, all_fin_detections.corrected_prediction)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')

print('With the found thresholds no filtering')
print('-------------------------------------------------')
cm = confusion_matrix(y_test, all_fin_detections.prediction)
print(f'Confusion Matrix: \n{cm}')
print(f'Area Under Curve: {roc_auc_score(y_test, all_fin_detections.prediction)}')
print(f'Recall or TPR score: {recall_score(y_test, all_fin_detections.prediction)}')
print(f'Precision score: {precision_score(y_test, all_fin_detections.prediction)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')

all_fin_detections.to_csv('./test/data/TkurtAll_Miller_thresholds_found_by_Clea_still_wrong.csv')


# ---------------------------------------------------------------------------------------
# Train models directly
noise_samples = all_fin_detections.loc[all_fin_detections.F20P == 0]
selected_noise_samples = noise_samples.sample(int(n_fin_detections * N_TIMES_BALANCED))
fin_samples = all_fin_detections.loc[all_fin_detections.F20P == 1]
balanced_dataset = pd.concat([fin_samples, selected_noise_samples])

# y_balanced = balanced_dataset['F20P']
# x_balanced = balanced_dataset[['Kurt', 'KurtProd', 'SNRF', 'SNRT', 'BW']]

y_all = all_fin_detections['F20P']
x_all = all_fin_detections[['Kurt', 'KurtProd', 'SNRF', 'SNRT', 'BW']]

x_train, x_test_balanced, y_train, y_test_balanced = train_test_split(x_all, y_all, test_size=0.2)

y_test = all_fin_detections.loc[~all_fin_detections.index.isin(x_train.index)]['F20P']
x_test = all_fin_detections.loc[~all_fin_detections.index.isin(x_train.index)][['Kurt',
                                                                               'KurtProd', 'SNRF', 'SNRT', 'BW']]

# RANDOM FOREST
param_distributions = {'criterion': ['gini', 'entropy'],
                       'min_samples_split': [2, 10, 20],
                       'min_samples_leaf': [1, 10, 20]}

rf = RandomForestClassifier(class_weight={0: 1, 1: N_TIMES_BALANCED})
random_searcher_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, cv=5, n_iter=50,
                                        scoring="f1")
random_searcher_rf.fit(x_train, y_train)
best_rf = random_searcher_rf.best_estimator_

# Evaluate the model in the test set
y_pred = best_rf.predict(x_test)

print('The RF results, trained on all data')
print('-------------------------------------------------')
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{cm}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall or TPR score: {recall_score(y_test,y_pred)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')
print(f'Precision score: {precision_score(y_test,y_pred)}')

# Plot ROC curves in the test set
plot_roc_curve(best_rf, y_test, x_test, 'RandomForst ROC test 0.2 split random')


# DECISION TREE ----------------------------------------------------------------------
dt = DecisionTreeClassifier(class_weight={0: 1, 1: N_TIMES_BALANCED}, max_depth=6)
random_searcher = RandomizedSearchCV(estimator=dt, param_distributions=param_distributions, cv=5, n_iter=50,
                                     scoring="f1")
random_searcher.fit(x_train, y_train)
best_dt = random_searcher.best_estimator_

# Evaluate the model in the test set
y_pred = best_dt.predict(x_test)

print('The DT results, trained on all data')
print('-------------------------------------------------')
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{cm}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall or TPR score: {recall_score(y_test,y_pred)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')
print(f'Precision score: {precision_score(y_test,y_pred)}')

# Plot ROC curves in the test set
plot_roc_curve(best_dt, y_test, x_test, 'DecisionTree ROC test 0.2 split random')

# # LOGISTIC REGRESSION ------------------------------------------------------------------
# # Create the model and train it
logi = LogisticRegression(class_weight={0: 1, 1: N_TIMES_BALANCED}, max_iter=100, penalty='l1', solver='liblinear')

logi.fit(x_train, y_train)

# Evaluate the model in the test set
y_pred = logi.predict(x_test)

print('The logistic regression results, trained on all data')
print('-------------------------------------------------')
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{cm}')
print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall or TPR score: {recall_score(y_test,y_pred)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')
print(f'Precision score: {precision_score(y_test,y_pred)}')

plot_roc_curve(logi, y_test, x_test, 'LogiCV ROC test 0.2 split random')

# LOGISTIC REGRESSION WITH CV -------------------------------------------------------
logicv = LogisticRegressionCV(class_weight={0: 1, 1: N_TIMES_BALANCED}, max_iter=100, penalty='l1', solver='liblinear')
logicv.fit(x_train, y_train)

# Evaluate the model in the test set
y_pred = logicv.predict(x_test)

print('The logistic regression CV results, trained on all data')
print('-------------------------------------------------')
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: \n{cm}')
print(f'Area Under Curve: {roc_auc_score(y_test, y_pred)}')
print(f'Recall or TPR score: {recall_score(y_test,y_pred)}')
print(f'FPR: {cm[0][1] / (cm[0][0] + cm[0][1])}')
print(f'Precision score: {precision_score(y_test,y_pred)}')

plot_roc_curve(logicv, y_test, x_test, 'LogiCV ROC test 0.2 split random')
