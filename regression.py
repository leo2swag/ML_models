from __future__ import division
import pandas as pd
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics

def regressionData(readinData, target_name, output):
    readinData = readinData.iloc[:, 1:]
    varNames = readinData.columns
    target = list(varNames).index(target_name)
    target_data = readinData.iloc[:,target]

    #print target_data.dtypes

    readinData.pop(target_name)

    train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(readinData, target_data, test_size=0.2,
                                                                            random_state=0)

    LR = LogisticRegression()
    LR.fit(train_data_X,train_data_Y)
    train_predict_data_Y = LR.predict(train_data_X)
    predict_data_Y = LR.predict(test_data_X)
    #print test_data_Y
    #print predict_data_Y
    #print(test_data_Y)
    test_data_Y = list(test_data_Y)
    #print readinData
    #print target_data
    predict_data_Y_prob = LR.predict_proba(test_data_X)
    print predict_data_Y_prob
    overall_acc = metrics.accuracy_score(test_data_Y,predict_data_Y)
    cm = confusion_matrix(test_data_Y,predict_data_Y)
    result_table = classification_report(test_data_Y,predict_data_Y)

    new_var_name = readinData.columns
    new_coef = LR.coef_
    resultdf = pd.DataFrame(columns=["Coef", "Variable"])
    for i in range(len(new_var_name)):
        temp = pd.DataFrame([[new_coef[0][i], new_var_name[i]]], columns=["Coef", "Variable"])
        resultdf = pd.concat([resultdf, temp], ignore_index=True)
    resultdf = pd.concat([resultdf, pd.DataFrame([[target_name, overall_acc]], columns=["Coef", "Variable"])])

    data_v2= pd.DataFrame(columns=["prob", "result"])
    #print test_data_Y[1][1]
    #print len(test_data_Y)
    #print len(predict_data_Y_prob)
    for i in range(len(predict_data_Y_prob)):

        temp = pd.DataFrame([[predict_data_Y_prob[i][1], test_data_Y[i]]], columns=["prob", "result"])
        data_v2 = pd.concat([data_v2, temp], ignore_index=True)

    print (overall_acc)
    print(cm)
    resultdf.to_excel(output, sheet_name='Coef')
    return data_v2


def AnalyzeData(data_v2, counter, output):

    analysedf = pd.DataFrame(columns=["PCT_flag", "contains1_1_num", "num","contains1_1_num_cum", "num_cum", "response", "response_acc", "consistency", "lift_value", "cover_rate", "cover_acc"])
    prob_df = data_v2['prob']
    total_1s = len(data_v2['result'][data_v2['result'] == 1])
    total_num = len(data_v2['result'])
    print total_num
    print(total_1s)
    k = 1 - 1/counter
    roundFirst = True
    pre = 0
    num_acc = 0
    contains_acc = 0
    response = 0
    response_acc = 0
    cover_rate = 0
    cover_acc = 0

    while (k >= 0):
       #print tt
        quant =  prob_df.quantile(k)
        #print quant
        if roundFirst:
            temp_data = data_v2[data_v2['prob'] >= quant]
            roundFirst = False
        else:
            temp_data = data_v2[(data_v2['prob'] >= quant) & (data_v2['prob'] < pre)]

        contains_1 = temp_data[temp_data['result'] == 1]
        num = len(temp_data)
        contains = len(contains_1)
        num_acc = num_acc + num
        contains_acc = contains_acc + contains

        cover_rate = contains / total_1s
        cover_acc = cover_acc + cover_rate

        temp_v3 = pd.DataFrame([[format(k, '.0%'), contains, num, contains_acc, num_acc, 0, 0, 0, 0, cover_rate, cover_acc]], columns=["PCT_flag", "contains1_1_num", "num","contains1_1_num_cum", "num_cum", "response", "response_acc", "consistency", "lift_value", "cover_rate", "cover_acc"])
        analysedf = pd.concat([analysedf, temp_v3], ignore_index=True)
        pre = quant

        k = float(round(k, 2)) - 1/counter

    analysedf['response'] = analysedf['contains1_1_num'] / analysedf['num']
    analysedf['response_acc'] = analysedf['contains1_1_num_cum'] / analysedf['num_cum']
    analysedf['consistency'] = total_1s / total_num
    analysedf['lift_value'] = analysedf['response'] / analysedf['consistency']
    #writer = pd.ExcelWriter(output)
    analysedf.to_excel(output, sheet_name='Analyze')
    #resultdf.to_excel(writer, sheet_name='Coef')
    #analysedf.to_excel(writer, sheet_name='Analyze')
    #writer.save()


def main():
    readinData = pd.read_excel("Data/excel_double11.xlsx")
    readinData['SEX_CDE'][readinData['SEX_CDE'] == 'MALE'] = 1
    readinData['SEX_CDE'][readinData['SEX_CDE'] == 'FEML'] = 0

    #print(readinData)
    #print(readinData.dtypes)
    #readinData['pid_num_m'][readinData['pid_num_m'] <= 150] = 0
    #readinData['pid_num_m'][readinData['pid_num_m'] > 150] = 1

    data_v2 = regressionData(readinData, 'flag_2015_target' , "results/regression_result_temp.xlsx")
    #print data_v2
    #AnalyzeData(data_v2, 20, "results/regression_result_analyze.xlsx")


if __name__=="__main__":
  main()