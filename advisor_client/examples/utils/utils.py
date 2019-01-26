def readModelConfigStructure(modelConfig):
    d = dict()
    for key in modelConfig.keys():
        if key == "train":
            for key2, value2 in modelConfig[key].items():
                if key2 == "params":
                    for key3,value3 in value2.items():
                        d[key3]= {key2:0}
                        d[key3][key2] = "train"
                    else:
                        d[key2] = "train"
                else:
                    d[key2] = "train"
        elif key =="normalize":
            for key2 in modelConfig[key].keys():
                d[key2] = "normalize"
        elif key =="stats":
            for key2 in modelConfig[key].keys():
                d[key2] = "stats"
        elif key =="varSelect":
            for key2 in modelConfig[key].keys():
                d[key2] = "varSelect"
    d["Loss"]="train"
    return d

def trainEvalSplit(inputFilePath, outputPath, testRatio,delimiter):
    import pandas as pd
    import numpy as np
    import subprocess
    import os
    # read data
    df = pd.read_csv(inputFilePath,delimiter=delimiter)
    # shuffle
    shuffledDF = df.sample(frac=1)
    # num of rows
    nRows = len(df)
    evalDF = df.iloc[:int(nRows * testRatio),:]
    trainDF = df.iloc[int(nRows * testRatio):,:]
    trainDF.to_csv(outputPath + "/trainData.csv",index = False,sep=delimiter)
    evalDF.to_csv(outputPath + "/evalData.csv",index = False,sep=delimiter)
    return outputPath + "/trainData.csv",outputPath + "/evalData.csv"

def evalMetrics(modelConfig,shifuJobPath,package,metricInfo):
    import json
    import subprocess
    metric = 0
    if package == "shifu":
        subprocess.call(["cd " + shifuJobPath + " && bash shifu eval > eval.log"],shell=True)
        if modelConfig["basic"]["runMode"] <> "LOCAL":
            subprocess.call(["rm " + shifuJobPath + "/evals/Eval1/EvalPerformance.json"],shell=True )
            subprocess.call(["cd " + shifuJobPath + "/evals/Eval1/" + " && hadoop fs -get " + str(modelConfig["basic"]["customPaths"])[1:-1] + "ModelSets/" + modelConfig["basic"]["name"] + "/evals/Eval1/EvalPerformance.json ." ],shell=True )
        with open(shifuJobPath + "/evals/Eval1/EvalPerformance.json") as f:
            evals = json.loads(f.read())
        if metricInfo == "auc":
            metric = evals["areaUnderRoc"]
        elif metricInfo.split(",")[1]=="actionRate":
            for op in evals["gains"]:
                if metricInfo.split(",")[0] == "catchRate":
                    if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                        metric = op["recall"]
                elif metricInfo.split(",")[0]=="hitRate":
                        if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                            metric = op["precision"]
                elif metricInfo.split(",")[0]=="weightedCatchRate":
                        if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                            metric = op["weightedRecall"]
                elif metricInfo.split(",")[0]=="weightedHitRate":
                        if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                            metric = op["weightedPrecision"]
        elif metricInfo.split(",")[1]=="weightedActionRate":
            for op in evals["gains"]:
                if metricInfo.split(",")[0] == "catchRate":
                    if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                        metric = op["recall"]
                elif metricInfo.split(",")[0]=="hitRate":
                        if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                            metric = op["precision"]
                elif metricInfo.split(",")[0]=="weightedCatchRate":
                        if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                            metric = op["weightedRecall"]
                elif metricInfo.split(",")[0]=="weightedHitRate":
                        if abs(op["actionRate"] - float(metricInfo.split(",")[2]))<10E-6:
                            metric = op["weightedPrecision"]
    return metric

def setDefaultTuningParams():
    import argparse
    parser = argparse.ArgumentParser()
    #stats
    parser.add_argument("-binningMethod", type=str, default="EqualPositive")
    parser.add_argument("-binningAlgorithm", type=str, default="SPDTI")
    #parser.add_argument("-binningAutoTypeEnable", type=str, default="false")
    #parser.add_argument("-binningAutoTypeThreshold", type=float, default="5")
    #parser.add_argument("-binningMergeEnable", type=str, default="true")
    parser.add_argument("-maxNumBin", type=int, default="10")
    parser.add_argument("-sampleRate", type=float, default="0.8")
    #parser.add_argument("-sampleNegOnly", type=str, default="false")
    #parser.add_argument("-numericalValueThreshold", type=float, default="1000000")
    #varsel
    parser.add_argument("-filterBy", type=str, default="IV")
    #parser.add_argument("-filterEnable", type=str, default="true")
    parser.add_argument("-filterNum", type=int, default="20")
    parser.add_argument("-filterOutRatio", type=float, default="0.05")
    #normalize
    parser.add_argument("-normType", type=str, default="WOE_ZSCALE")
    #train
    #parser.add_argument("-baggingNum", type=float, default="1")
    #parser.add_argument("-baggingWithReplacement", type=str, default="false")
    #parser.add_argument("-baggingSampleRate", type=float, default="0.8")
    #parser.add_argument("-validSetRate", type=float, default="0.2")
    #parser.add_argument("-algorithm", type=str, default="NN")
    parser.add_argument("-Package", type=str, default="shifu")
    parser.add_argument("-Loss",type=str,default = "Squared")
    # NN
    parser.add_argument("-LearningRate", type=float, default="0.1")
    parser.add_argument("-ActivationFunc", type=str, default="ReLU")
    parser.add_argument("-NumHiddenLayers", type=int, default="1")
    parser.add_argument("-NumHiddenNodes", type=int, default="10")
    parser.add_argument("-numTrainEpochs",type=int,default="100")
    # GBT
    #parser.add_argument("-MaxDepth", type=float, default=10)
    #parser.add_argument("-TreeNum", type=float, default=10)
    return parser

def setDefaultParams():
    import argparse
    parser = setDefaultTuningParams()
    #info
    parser.add_argument("-trialID",type=str,default = "0")
    parser.add_argument("-studyName",type=str,default = "shifuJob")
    parser.add_argument("-metricInfo",type=str,default = "catchRate,actionRate,0.02")
    parser.add_argument("-trainDataPath",type=str,default = "../data/givemesomecredit/cs-training.csv")
    parser.add_argument("-evalDataPath",type=str,default = "")
    return parser

def setParamsHadoop():
    import argparse
    parser =  setDefaultTuningParams()
    #job
    parser.add_argument("-trialID",type=str,default = "0")
    parser.add_argument("-studyName",type=str,default = "shifuJob")
    parser.add_argument("-metricInfo",type=str,default = "catchRate,actionRate,0.02")
    #data
    parser.add_argument("-targetColumnName",type=str,default = "")
    parser.add_argument("-negTags",type=str,default = ["0"])
    parser.add_argument("-posTags",type=str,default = ["1"])
    parser.add_argument("-categoricalColumnNameFile",type=str,default = "")
    parser.add_argument("-metaColumnNameFile",type=str,default = "")
    parser.add_argument("-dataDelimiter",type=str,default = ",")
    parser.add_argument("-dataHeader",type=str,default = "")
    # train data
    parser.add_argument("-trainDataPath",type=str,default = "")
    parser.add_argument("-trainDataWeightColumnName",type=str,default = "")
    parser.add_argument("-trainDataFilterExpressions",type=str,default = "")
    # eval data
    parser.add_argument("-evalDataPath",type=str,default = "")
    parser.add_argument("-evalDataWeightColumnName",type=str,default = "")
    parser.add_argument("-evalDataFilterExpressions",type=str,default = "")
    parser.add_argument("-performanceBucketNum",type=str,default = "")
    return parser
