#coding:utf-8
from __future__ import division
import pandas as pd
import mord as m
import random
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.feature_selection import chi2
import datetime


def regressionData(readinData, target_name, output):
    readinData = readinData.iloc[:, 1:]
    convert = 24*60*60*365*1000000000

    #readinData['Fst_ACCT_OPEN_DT'] = datetime.datetime(readinData['Fst_ACCT_OPEN_DT'].astype(str).split('-'))
    readinData['Fst_ACCT_OPEN_DT'] = (datetime.datetime(2017,12,31) - readinData['Fst_ACCT_OPEN_DT'])
    #print readinData['Fst_ACCT_OPEN_DT']
    readinData['Fst_ACCT_OPEN_DT'] = readinData['Fst_ACCT_OPEN_DT'].astype(np.int64)
    #print readinData['Fst_ACCT_OPEN_DT']
    readinData['Fst_ACCT_OPEN_DT'] = pd.to_numeric(readinData['Fst_ACCT_OPEN_DT'] / convert)
    #print readinData['Fst_ACCT_OPEN_DT']

    varNames = readinData.columns
    target = list(varNames).index(target_name)
    target_data = readinData.iloc[:, target]
    #print target_data
    readinData.pop(target_name)
    #print readinData.dtypes
    print "Ready for the model"
    train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(readinData, target_data, test_size=0.3,
                                                                            random_state=0)

    train_data_X = train_data_X.astype(np.float64)
    test_data_X = test_data_X.astype(np.float64)
    #print train_data_Y
    #print train_data_X.dtypes

    train_data_Y = train_data_Y.astype(np.int64)
    test_data_Y = test_data_Y.astype(np.int64)
    print train_data_Y.dtypes
    print test_data_Y.dtypes
    print train_data_X.dtypes
    print test_data_X.dtypes

    #print train_data_X
    LR = m.LogisticIT().fit(train_data_X, train_data_Y)
    #LR.fit(train_data_X, train_data_Y)
    predict_data_Y = LR.predict(test_data_X)
    print "finish predicting"
    #print test_data_X

    #predict_data_Y_prob = LR.predict_proba(test_data_X)
    overall_acc = metrics.accuracy_score(test_data_Y, predict_data_Y)
    print overall_acc
    #cm = confusion_matrix(test_data_Y, predict_data_Y)
    #result_table = classification_report(test_data_Y, predict_data_Y)

    readinData = readinData.rename(
        columns={'Gender_Cd': '性别', 'Age': '年龄', 'Fst_ACCT_OPEN_DT':'开户时长','clu73': '活期存款业务活跃度','Is_PP_Cust': '是否开通手机贴膜卡业务','Is_EP_Cust':'是否开通第三方支付业务','clu19': '是否持有信用卡', 'clu20': '是否持有借记卡', 'clu21': '是否持有存折', 'clu212': '是否持有存单', 'clu213': '是否持有定期一本通',
                 'Is_INSU_Cust':'是否社保客户','clu214': '是否持有活期一本通', 'AUM_0_5': 'AUM资产在0至5万之间客户数',
                 'Is_DFDK_CARD_Cust':'代发客户是否持有卡','clu37': '持有定期产品数量', 'clu38': '持有大额存单数量', 'clu39': '理财产品数量', 'clu40': '基金产品数量', 'clu41': '贵金属产品数量',
                 'Is_DFDK_CZ_Cust':'代发客户是否持有存折','clu42': '信托产品数量', 'clu43': '代销储蓄国债产品数量',
                 'clu44': '代理保险产品数量', 'clu45': '银证第三方存管产品数量', 'clu46': '个人消费贷款产品数量', 'clu47': '个人经营贷款产品数量',
                 'clu471': '个人委托贷款产品数量', 'clu48': '信用卡数量', 'clu72': '定期存款业务活跃度',
                 'clu74': '贷款业务活跃度', 'clu75': '理财业务活跃度', 'Is_NW_Cust': '是否开通网上银行业务', 'Is_PB_Cust': '是否开通手机银行业务',
                 'Is_WE_Cust': '是否开通微信银行业务', 'Is_DFDK_Cust': '是否代发客户', 'CB_CT_TX_NUM': '核心客户柜面使用频率',
                 'CB_PB_TX_NUM': '核心客户手机银行使用频率', 'CB_PP_TX_NUM': '核心客户手机贴膜卡使用频率',
                 'CB_NW_TX_NUM': '核心客户网上银行使用频率', 'CB_WE_TX_NUM': '核心客户微信银行使用频率(非动帐)', 'CB_ATM_TX_NUM': '核心客户ATM使用频率',
                 'CB_EP_TX_NUM': '核心客户第三方支付平台使用频率', 'CB_POS_TX_NUM': '核心客户POS/TPOS使用频率',
                 'indicator_new': '是否过路资金账户'})

    new_var_name = readinData.columns
    new_coef = LR.coef_

    scores,pvalues = chi2(train_data_X,train_data_Y)

    print "Start writing"
    resultdf = pd.DataFrame(columns=["Coef", "Variable", "pvalue"])
    for i in range(len(new_var_name)):
        temp = pd.DataFrame([[new_coef[i], new_var_name[i], pvalues[i]]], columns=["Coef", "Variable", "pvalue"])
        resultdf = pd.concat([resultdf, temp], ignore_index=True)
    resultdf = pd.concat([resultdf, pd.DataFrame([[target_name, overall_acc, 1]], columns=["Coef", "Variable", "pvalue"])])
    
    #data_v2= pd.DataFrame(columns=["prob", "result"])
    #print test_data_Y[1][1]
    #print len(test_data_Y)
    #print len(predict_data_Y_prob)
    #for i in range(len(predict_data_Y_prob)):

    #    temp = pd.DataFrame([[predict_data_Y_prob[i], test_data_Y[i]]], columns=["prob", "result"])
    #    data_v2 = pd.concat([data_v2, temp], ignore_index=True)

    resultdf.to_csv(output, encoding='utf_8_sig', index=False)
    print "Done"





def main():
    #readinData = pd.read_sas("Data/sample_cust.sas7bdat")
    readinData = pd.read_sas("Data/cust_info_model.sas7bdat")
    #readinData = pd.read_excel("Data/pre_model.xls")
    readinData['Gender_Cd'][readinData['Gender_Cd'] == 'M'] = 1
    readinData['Gender_Cd'][readinData['Gender_Cd'] == 'F'] = 0
    readinData['Gender_Cd'][readinData['Gender_Cd'] == 'U'] = np.nan
    readinData = readinData.dropna(axis=0, how='any')
    readinData['Gender_Cd'] = pd.to_numeric(readinData['Gender_Cd'])
    #print readinData.head

    readinData['clum28'][(readinData['clum28'] >= 0) & (readinData['clum28'] < 50000)] = 1
    readinData['clum28'][(readinData['clum28'] >= 50000) & (readinData['clum28'] < 200000)] = 2
    readinData['clum28'][(readinData['clum28'] >= 200000) & (readinData['clum28'] < 1000000)] = 3
    readinData['clum28'][readinData['clum28'] >= 1000000] = 4

    # print(readinData.dtypes)
    # readinData['pid_num_m'][readinData['pid_num_m'] <= 150] = 0
    # readinData['pid_num_m'][readinData['pid_num_m'] > 150] = 1

    regressionData(readinData, 'clum28', "results/regression_result_multi_total.csv")
    # print data_v2
    # AnalyzeData(data_v2, 20, "results/regression_result_analyze.xlsx")


if __name__ == "__main__":
    main()