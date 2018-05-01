import numpy as np
import xlrd
import xlwt
from sklearn.cluster import KMeans

def openFunction(fileName):
    data = (xlrd.open_workbook(fileName)).sheets()[0]
    nrows = data.nrows
    list = []
    for i in range(nrows):
        colList = data.row_values(i)
        list.append(colList)
    vList = list[0]
    vList.append("Cluster_Label")
    list.pop(0)
    print "datalist is \n" + str(list)
    return list,vList


def removeFunction(sortByCol,removedEle, variablelist):
    newcol = []
    newname = []
    newcolSize = len(sortByCol)
    det = 1
    while det <= newcolSize:
        if str(det) in removedEle:
            det = det + 1
        else:
            newcol.append(sortByCol[det - 1])
            newname.append(variablelist[det-1])
            det = det + 1
    newname.append("Cluster_Label")
    newname[0] = "Cluster_Result"
    newcol.pop(0)
    newrol = [list(i) for i in zip(*newcol)]
    return newrol,newname,newcol


def standardFunction(fixedVersion):
    output = []
    for i in range(len(fixedVersion)):
        allZeros = False
        minValue = min(fixedVersion[i])
        maxValue = max(fixedVersion[i])
        templen = len(fixedVersion[i])
        if maxValue == 0 and minValue == 0:
            allZeros = True
        outcol = []
        arr = fixedVersion[i]
        for k in range(templen):
            if allZeros:
                value = 0
            else:
                value = round(((arr[k] - minValue) / (maxValue - minValue)), 4)
            outcol.append(value)
        output.append(outcol)
    newoutput = [list(i) for i in zip(*output)]
    return newoutput


def plotData(fixedVersion,oriVersion, variableList,oldVari,standardVersion,n,outputfile):
    clf = KMeans(n_clusters=n)
    y_pred = clf.fit_predict(standardVersion)
    clusterList = [[]]
    for i in range(n):
        eachCluster = []
        clusterList.append(eachCluster)
    yLen = len(y_pred)
    yCounter = 0
    while yCounter < yLen:
        ypredValue = y_pred[yCounter]
        oriVersion[yCounter].append(int(ypredValue)+1)
        clusterList[ypredValue].append(fixedVersion[yCounter])
        yCounter = yCounter + 1
    nCounter = 0
    clusterMeans = [[]]
    for q in range(n):
        calCluster = []
        clusterMeans.append(calCluster)
    while nCounter < n:
        newclu = [list(i) for i in zip(*clusterList[nCounter])]
        newcluLen = len(newclu)
        for i in range(newcluLen):
            clumeans = round(np.mean(newclu[i]),2)
            clusterMeans[nCounter].append(clumeans)
        clusterMeans[nCounter].append(nCounter+1)
        nCounter = nCounter + 1
    workbook = xlwt.Workbook(encoding="utf-8")
    data_sheet = workbook.add_sheet("cluster_" + str(n))

    pos = 0
    for k in range(n):
        data_sheet.write(k,0,"Section " + str(k+1) + " has " + str(len(clusterList[k])))
        pos = k
    pos = pos + 3
    for k in range(len(oldVari)):
        data_sheet.write(pos, k, oldVari[k])
    pos = pos + 2
    temppos = pos
    for k in range(len(oriVersion)):
        temprow = oriVersion[k]
        for w in range(len(temprow)):
            data_sheet.write(pos+k,w,temprow[w])
            temppos = pos+k
    pos = temppos + 3
    for k in range(len(variableList)):
        data_sheet.write(pos, k, variableList[k])
    pos = pos + 2
    for k in range(len(clusterMeans)):
        tempcluster = clusterMeans[k]
        for w in range(len(tempcluster)):
            data_sheet.write(pos+k,w+1,tempcluster[w])
    workbook.save(outputfile)


def main():
    fileName = raw_input("Which file you would like to read in ?\n")
    clusterNumber = raw_input("How many groups you want for clustering?\n")
    removedEle = raw_input("Which variables you would like to exclude for clustering?\n")
    outputfile = raw_input("What you would like to output?\n")
    try:
        tableName = openFunction(fileName)[0]
    except:
        print "\nError in opening files, Please pass another file to read in\n"
    variableList = openFunction(fileName)[1]
    sortByCol = [list(i) for i in zip(*tableName)]
    fixedVersion = removeFunction(sortByCol,removedEle,variableList)[0]
    newvariablelist = removeFunction(sortByCol,removedEle,variableList)[1]
    standardVersion = standardFunction(removeFunction(sortByCol, removedEle, variableList)[2])
    plotData(fixedVersion,tableName,newvariablelist,variableList,standardVersion,int(clusterNumber),outputfile)

    print("\n################################################")
    print("Successfully outputing the file with cluster")
    print("################################################\n")


if __name__ == "__main__":
    main()
