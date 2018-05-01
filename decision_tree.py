#coding:utf-8
from __future__ import division
import pandas as pd
import mord as m
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import datetime
import pydotplus
import collections

def decisionTree(readinData, target_name, output):
    readinData = readinData.iloc[:, 1:]
    convert = 24 * 60 * 60 * 365 * 1000000000

    # readinData['Fst_ACCT_OPEN_DT'] = datetime.datetime(readinData['Fst_ACCT_OPEN_DT'].astype(str).split('-'))
    readinData['Fst_ACCT_OPEN_DT'] = (datetime.datetime(2017, 12, 31) - readinData['Fst_ACCT_OPEN_DT'])
    # print readinData['Fst_ACCT_OPEN_DT']
    readinData['Fst_ACCT_OPEN_DT'] = readinData['Fst_ACCT_OPEN_DT'].astype(np.int64)
    # print readinData['Fst_ACCT_OPEN_DT']
    readinData['Fst_ACCT_OPEN_DT'] = pd.to_numeric(readinData['Fst_ACCT_OPEN_DT'] / convert)

    varNames = readinData.columns
    target = list(varNames).index(target_name)
    target_data = readinData.iloc[:, target]
    #print target_data
    readinData.pop(target_name)

    train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(readinData, target_data, test_size=0.3,
                                                                            random_state=0)

    train_data_X = train_data_X.astype(np.float64)
    test_data_X = test_data_X.astype(np.float64)
    train_data_Y = train_data_Y.astype(np.int64)
    test_data_Y = test_data_Y.astype(np.int64)

    dt = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20)
    dt.fit(train_data_X, train_data_Y)
    #print test_data_X
    #predict_data_Y = DecisionTreeClassifier.predict_proba(test_data_X)
    #print predict_data_Y
    #overall_acc = metrics.accuracy_score(test_data_Y, predict_data_Y)
    score_value = dt.score(test_data_X, test_data_Y)
    print score_value
    #tree.export_graphviz(dt,out_file='tree_less.dot')

    print "Starts to generate trees png"
    tree.export_graphviz(dt,
                                    out_file='results/tree_more.dot',
                                    filled=True,
                                    rounded=True)


    print "Tree png generated, starts to make excel files"
    readinData = readinData.rename(
        columns={'Gender_Cd': '性别', 'Age': '年龄', 'Fst_ACCT_OPEN_DT': '开户时长', 'clu73': '活期存款业务活跃度',
                 'Is_PP_Cust': '是否开通手机贴膜卡业务', 'Is_EP_Cust': '是否开通第三方支付业务', 'clu19': '是否持有信用卡', 'clu20': '是否持有借记卡',
                 'clu21': '是否持有存折', 'clu212': '是否持有存单', 'clu213': '是否持有定期一本通',
                 'Is_INSU_Cust': '是否社保客户', 'clu214': '是否持有活期一本通', 'AUM_0_5': 'AUM资产在0至5万之间客户数',
                 'Is_DFDK_CARD_Cust': '代发客户是否持有卡', 'clu37': '持有定期产品数量', 'clu38': '持有大额存单数量', 'clu39': '理财产品数量',
                 'clu40': '基金产品数量', 'clu41': '贵金属产品数量',
                 'Is_DFDK_CZ_Cust': '代发客户是否持有存折', 'clu42': '信托产品数量', 'clu43': '代销储蓄国债产品数量',
                 'clu44': '代理保险产品数量', 'clu45': '银证第三方存管产品数量', 'clu46': '个人消费贷款产品数量', 'clu47': '个人经营贷款产品数量',
                 'clu471': '个人委托贷款产品数量', 'clu48': '信用卡数量', 'clu72': '定期存款业务活跃度',
                 'clu74': '贷款业务活跃度', 'clu75': '理财业务活跃度', 'Is_NW_Cust': '是否开通网上银行业务', 'Is_PB_Cust': '是否开通手机银行业务',
                 'Is_WE_Cust': '是否开通微信银行业务', 'Is_DFDK_Cust': '是否代发客户', 'CB_CT_TX_NUM': '核心客户柜面使用频率',
                 'CB_PB_TX_NUM': '核心客户手机银行使用频率', 'CB_PP_TX_NUM': '核心客户手机贴膜卡使用频率',
                 'CB_NW_TX_NUM': '核心客户网上银行使用频率', 'CB_WE_TX_NUM': '核心客户微信银行使用频率(非动帐)', 'CB_ATM_TX_NUM': '核心客户ATM使用频率',
                 'CB_EP_TX_NUM': '核心客户第三方支付平台使用频率', 'CB_POS_TX_NUM': '核心客户POS/TPOS使用频率',
                 'indicator_new': '是否过路资金账户'})
    varNames = readinData.columns
    check = pd.DataFrame(columns=['index', 'variableName'])
    for i in range(len(varNames)):
        temp = pd.DataFrame([[i, varNames[i]]],columns=['index', 'variableName'])
        check = pd.concat([check, temp], ignore_index=True)
    #check = pd.concat([check, pd.DataFrame([['Accuracy', overall_acc]], columns=['index, variableName'])], ignore_index=True)
    check.to_csv('results/checkVari.csv',encoding='utf_8_sig', index=False)
    #LR.fit(train_data_X, train_data_Y)
    #predict_data_Y = dt.predict(test_data_X)
    #predict_data_Y = [0,4,2]
    #print test_data_X


def main():
    readinData = pd.read_sas("Data/cust_info_model.sas7bdat")
    #readinData = pd.read_sas("Data/sample_cust.sas7bdat")
    #readinData = pd.read_excel("Data/pre_model.xls")
    readinData['Gender_Cd'][readinData['Gender_Cd'] == 'M'] = 1
    readinData['Gender_Cd'][readinData['Gender_Cd'] == 'F'] = 0
    readinData['Gender_Cd'][readinData['Gender_Cd'] == 'U'] = np.nan
    readinData = readinData.dropna(axis=0, how='any')
    readinData['Gender_Cd'] = pd.to_numeric(readinData['Gender_Cd'])
    # print readinData.head

    readinData['clum28'][(readinData['clum28'] >= 0) & (readinData['clum28'] < 50000)] = 1
    readinData['clum28'][(readinData['clum28'] >= 50000) & (readinData['clum28'] < 200000)] = 2
    readinData['clum28'][(readinData['clum28'] >= 200000) & (readinData['clum28'] < 1000000)] = 3
    readinData['clum28'][readinData['clum28'] >= 1000000] = 4
    # print(readinData)
    # print(readinData.dtypes)
    # readinData['pid_num_m'][readinData['pid_num_m'] <= 150] = 0
    # readinData['pid_num_m'][readinData['pid_num_m'] > 150] = 1

    decisionTree(readinData, 'clum28', "results/regression_result_dt.xlsx")


if __name__ == "__main__":
    main()