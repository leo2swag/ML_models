#coding:utf-8
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import codecs






def main():
    
    readinData = pd.read_sas("Data/cust_info_model.sas7bdat")
    #print len(readinData)
    #print readinData.columns
    #dict = {}
    print readinData
    readinData.pop('Cust_Id')
    '''
    readinData = readinData[(readinData['Gender_Cd'] == 'M') | (readinData['Gender_Cd'] == 'F')]
    readinData['guy'] = readinData['Age'].copy(deep=True)
    readinData['girl'] = readinData['Age'].copy(deep=True)
    readinData['guy'][readinData['Gender_Cd'] == 'M'] = 1
    readinData['guy'][readinData['Gender_Cd'] == 'F'] = 0
    readinData['girl'][readinData['Gender_Cd'] == 'F'] = 1
    readinData['girl'][readinData['Gender_Cd'] == 'M'] = 0

    
    readinData['Age_under_30'] = readinData['Age'].copy(deep=True)
    readinData['Age_under_30'][readinData['Age'] < 30] = 1
    readinData['Age_under_30'][readinData['Age'] >= 30] = 0

    readinData['Age_30_65'] = readinData['Age'].copy(deep=True)
    readinData['Age_30_65'][readinData['Age'] < 30] = 0
    readinData['Age_30_65'][(readinData['Age'] >= 30) & (readinData['Age'] <= 65)] = 1
    readinData['Age_30_65'][readinData['Age'] > 65] = 0

    readinData['Age_65_above'] = readinData['Age'].copy(deep=True)
    readinData['Age_65_above'][readinData['Age'] <= 65] = 0
    readinData['Age_65_above'][readinData['Age'] > 65] = 1

    readinData['Year_use'] = readinData['Fst_ACCT_OPEN_DT'].copy(deep=True)
    readinData['Year_use'] = pd.to_numeric(readinData['Year_use'].astype(str).str[:4])

    readinData['Year_use_under_1'] = readinData['Year_use'].copy(deep=True)
    readinData['Year_use_under_1'][readinData['Year_use_under_1'] < 2017] = 0
    readinData['Year_use_under_1'][readinData['Year_use_under_1'] >= 2017] = 1

    readinData['Year_use_1_2'] = readinData['Year_use'].copy(deep=True)
    readinData['Year_use_1_2'][readinData['Year_use_1_2'] < 2016] = 0
    readinData['Year_use_1_2'][(readinData['Year_use_1_2'] >= 2016) & (readinData['Year_use_1_2'] < 2017)] = 1
    readinData['Year_use_1_2'][readinData['Year_use_1_2'] >= 2017] = 0

    readinData['Year_use_3_5'] = readinData['Year_use'].copy(deep=True)
    readinData['Year_use_3_5'][readinData['Year_use_3_5'] < 2013] = 0
    readinData['Year_use_3_5'][(readinData['Year_use_3_5'] >= 2013) & (readinData['Year_use_3_5'] < 2016)] = 1
    readinData['Year_use_3_5'][readinData['Year_use_3_5'] >= 2016] = 0

    readinData['Year_use_5_above'] = readinData['Year_use'].copy(deep=True)
    readinData['Year_use_5_above'][readinData['Year_use_5_above'] < 2013] = 1
    readinData['Year_use_5_above'][readinData['Year_use_5_above'] >= 2013] = 0

    readinData["AUM_0_5"] = readinData['clum28'].copy(deep=True)
    readinData['AUM_0_5'][readinData['AUM_0_5'] <= 0] = 0
    readinData['AUM_0_5'][(readinData['AUM_0_5'] <= 5) & (readinData['AUM_0_5'] > 0)] = 1
    readinData['AUM_0_5'][readinData['AUM_0_5'] > 5] = 0

    readinData["AUM_5_20"] = readinData['clum28'].copy(deep=True)
    readinData['AUM_5_20'][readinData['AUM_5_20'] <= 5] = 0
    readinData['AUM_5_20'][(readinData['AUM_5_20'] <= 20) & (readinData['AUM_5_20'] > 5)] = 1
    readinData['AUM_5_20'][readinData['AUM_5_20'] > 20] = 0

    readinData["AUM_20_100"] = readinData['clum28'].copy(deep=True)
    readinData['AUM_20_100'][readinData['AUM_20_100'] <= 20] = 0
    readinData['AUM_20_100'][(readinData['AUM_20_100'] <= 100) & (readinData['AUM_20_100'] > 20)] = 1
    readinData['AUM_20_100'][readinData['AUM_20_100'] > 100] = 0

    readinData["AUM_above_100"] = readinData['clum28'].copy(deep=True)
    readinData['AUM_above_100'][readinData['AUM_above_100'] <= 100] = 0
    readinData['AUM_above_100'][readinData['AUM_above_100'] > 100] = 1
    '''

    dlist = ["Is_NW_Cust", "Is_PB_Cust", "Is_WE_Cust", "Is_DFDK_Cust", "clu19", "clu20", "clu21",
    "clu212", "clu213", "clu214", "indicator_new"]
    ldlen = len(dlist)
    for i in range(ldlen):
        dname = dlist[i]
        readinData[dname] = pd.to_numeric(readinData[dname])

    list = ["clu72","clu73","clu74","clu75", "CB_CT_TX_NUM","CB_PB_TX_NUM","CB_PP_TX_NUM","CB_NW_TX_NUM", "CB_WE_TX_NUM", "CB_ATM_TX_NUM", "CB_EP_TX_NUM", "CB_POS_TX_NUM"]
    llen = len(list)
    for i in range(llen):
        name = list[i]
        #new_name = "New_" + name
        #print new_name
        median = readinData[name].quantile(0.5)
        readinData[dname] = readinData[dname].fillna(median)
        #print median
        #readinData[name] = readinData[name].copy(deep=True)
        readinData[name][readinData[name] <= median] = 0
        readinData[name][readinData[name] > median] = 1

    alist = ["clu37","clu38","clu39","clu40","clu41","clu42","clu43","clu44","clu45","clu46","clu47","clu471","clu48"]
    alen = len(alist)
    for i in range(alen):
        aname = alist[i]
        # new_name = "New_" + name
        # print new_name
        readinData[aname] = readinData[aname].fillna(0)
        # print median
        # readinData[name] = readinData[name].copy(deep=True)
        readinData[aname][readinData[aname] <= 0] = 0
        readinData[aname][readinData[aname] > 0] = 1

    readinData.pop('Age')
    readinData.pop('Fst_ACCT_OPEN_DT')
    readinData.pop('clum28')
    #readinData.pop('Year_use')
    readinData.pop('Gender_Cd')
    readinData.pop('Is_PP_Cust')
    readinData.pop('clu73')
    readinData.pop('clu20')

    readinData = readinData.rename(columns={'clu19':'是否持有信用卡', 'clu20':'是否持有借记卡', 'clu21':'是否持有存折', 'clu212':'是否持有存单', 'clu213':'是否持有定期一本通', 'clu214':'是否持有活期一本通', 'AUM_0_5':'AUM资产在0至5万之间客户数',
                               'clu37':'持有定期产品数量', 'clu38':'持有大额存单数量', 'clu39':'理财产品数量', 'clu40':'基金产品数量', 'clu41':'贵金属产品数量', 'clu42':'信托产品数量', 'clu43':'代销储蓄国债产品数量',
                                'clu44':'代理保险产品数量', 'clu45':'银证第三方存管产品数量', 'clu46':'个人消费贷款产品数量', 'clu47':'个人经营贷款产品数量', 'clu471':'个人委托贷款产品数量', 'clu48':'信用卡数量', 'clu72':'定期存款业务活跃度',
                                'clu74':'贷款业务活跃度', 'clu75':'理财业务活跃度', 'Is_NW_Cust':'是否开通网上银行业务', 'Is_PB_Cust':'是否开通手机银行业务',
                               'Is_WE_Cust':'是否开通微信银行业务', 'Is_DFDK_Cust':'是否代发客户', 'CB_CT_TX_NUM':'核心客户柜面使用频率', 'CB_PB_TX_NUM':'核心客户手机银行使用频率', 'CB_PP_TX_NUM':'核心客户手机贴膜卡使用频率',
                               'CB_NW_TX_NUM':'核心客户网上银行使用频率', 'CB_WE_TX_NUM':'核心客户微信银行使用频率(非动帐)', 'CB_ATM_TX_NUM':'核心客户ATM使用频率', 'CB_EP_TX_NUM':'核心客户第三方支付平台使用频率', 'CB_POS_TX_NUM':'核心客户POS/TPOS使用频率',
                               'indicator_new':'是否过路资金账户'})
    #print readinData

    #print readinData.columns
    
    frequent_itemsets = apriori(readinData, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    rules = rules.sort_values(['confidence'], ascending=False)
    print(rules)
    #print rules.head()

    #writer = pd.ExcelWriter("results/tianjin.xlsx")
    #fos.write(codecs.BOM_UTF8)
    #rules.write()
    rules.to_csv('results/tianjin.csv', encoding='utf_8_sig', index=False)
    #rules.to_excel("results/tianjin.xlsx", encoding="utf-8", index=False)
    #writer.save()
    

if __name__=="__main__":
  main()

  # if response = 0.89, meaning every 10 inputs, we can predict 9 correct one
  # 模型中每预测出10个1，能确保8.9个是正确的1
  # if cover_rate = 0.1 for 95% meanign every first 5% of the data can cover 10% 1
  # 在前百分之五的数据里面，预测出1的概率占全部数据的百分之十，两者都是越大越好 意味着能用更少的人力来获得近可能多的资源
  # print data_v2