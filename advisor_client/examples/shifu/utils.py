def readModelConfigStructure(modelConfig):
    d = dict()
    for key in modelConfig.keys():
        if key == 'train':
            for key2, value2 in modelConfig[key].items():
                if key2 == "params":
                    for key3,value3 in value2.items():
                        d[key3]= {key2:0}
                        d[key3][key2] = "train"
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
def evalMetrics(evalConfig,shifuJobPath,package,metricType):
    import json
    import subprocess
    if package == "shifu":
        subprocess.call(["cd " + shifuJobPath + " && bash shifu eval > eval.log"],shell=True)
        if metricType == "auc":
            with open(shifuJobPath + "/evals/Eval1/EvalPerformance.json") as f:
                evals = json.loads(f.read())
            metric = evals["areaUnderRoc"]
    return metric
