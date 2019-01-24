#!/usr/bin/env python

import argparse
import subprocess
import json
import os
import sys
sys.path.append("..")
from utils.utils import readModelConfigStructure,trainEvalSplit,evalMetrics,setDefaultParams
import pandas as np

args = setDefaultParams().parse_args()

def main():

    #get optimzation job info
    studyName=vars(args)["studyName"]
    trialID=vars(args)["trialID"]
    trainDataPath=vars(args)["trainDataPath"]
    evalDataPath = vars(args)["evalDataPath"]
    metricInfo=vars(args)["metricInfo"]
    del args.studyName
    del args.trialID
    del args.trainDataPath
    del args.evalDataPath
    del args.metricInfo
    del args.Package

    # get path
    currentPath = os.getcwd()

    # create a shifu job
    shifuJobName = studyName
    #shifu job path
    shifuJobPath = currentPath + "/" + shifuJobName
    subprocess.call(["cd " + currentPath + " && shifu new " + shifuJobName],shell=True)
    if not os.path.isdir( shifuJobPath + "/modelLibrary"):
        subprocess.call(["mkdir " + shifuJobPath + "/modelLibrary"],shell=True)

    # split data to train and test
    if evalDataPath == "":
        if not os.path.exists(shifuJobPath + '/evalData.csv') or not os.path.exists(shifuJobPath + '/evalData.csv'):
            trainDataPath, evalDataPath = trainEvalSplit(trainDataPath, shifuJobPath, 0.2,",")
        else:
            trainDataPath = shifuJobPath + '/trainData.csv'
            evalDataPath = shifuJobPath + '/evalData.csv'
    else:
        if not os.path.exists(shifuJobPath + '/evalData.csv') or not os.path.exists(shifuJobPath + '/evalData.csv'):
            subprocess.call(["cp " + trainDataPath + " " + shifuJobPath + '/trainData.csv'],shell=True)
            subprocess.call(["cp " + evalDataPath +  " " + shifuJobPath + '/evalData.csv'],shell=True)
            trainDataPath = shifuJobPath + '/trainData.csv'
            evalDataPath = shifuJobPath + '/evalData.csv'

    # Change model config
    with open(shifuJobPath + "/ModelConfig.json","r+") as modelConfigFile:
        modelConfig = json.loads(modelConfigFile.read())
        modelConfigStructure = readModelConfigStructure(modelConfig)

        # change job info
        # train data
        modelConfig["dataSet"]["dataPath"] = trainDataPath
        modelConfig["dataSet"]["dataDelimiter"] = ","
        modelConfig["dataSet"]["headerPath"] = ""
        modelConfig["dataSet"]["headerDelimiter"] = ","
        modelConfig["dataSet"]["targetColumnName"] = "SeriousDlqin2yrs"
        modelConfig["dataSet"]["negTags"] = ["0"]
        modelConfig["dataSet"]["posTags"] = ["1"]

        # eval data
        modelConfig["evals"][0]["dataSet"]["dataPath"] =  evalDataPath
        modelConfig["evals"][0]["dataSet"]["dataDelimiter"] = ","
        modelConfig["evals"][0]["dataSet"]["headerPath"] = ""
        modelConfig["evals"][0]["dataSet"]["headerDelimiter"] = ","
        modelConfig["evals"][0]["dataSet"]["targetColumnName"] = "SeriousDlqin2yrs"
        modelConfig["evals"][0]["dataSet"]["negTags"] = ["0"]
        modelConfig["evals"][0]["dataSet"]["posTags"] = ["1"]
        modelConfig["evals"][0]["performanceBucketNum"] = "100"

        # training setting
        # change optimizing target parameters

        # assign arguements
        for arg, value in vars(args).items():
            # stats
            if modelConfigStructure[arg] == "stats":
                modelConfig["stats"][arg] = value

            # normalize
            elif modelConfigStructure[arg] == "normalize":
                modelConfig["normalize"][arg] = value

            # varSelect
            elif modelConfigStructure[arg] == "varSelect":
                modelConfig["varSelect"][arg] = value

            # train
            elif modelConfigStructure[arg] == "train":
                modelConfig["train"][arg] = value
            else:
                if arg == "NumHiddenNodes" or arg == "ActivationFunc":
                    modelConfig["train"]["params"][arg] = [value]
                else:
                    modelConfig["train"]["params"][arg] = value

    with open(shifuJobPath + "/ModelConfig.json", "wt") as file:
        json.dump(modelConfig,file,indent=2,sort_keys=True)

    # add meta column names
    subprocess.call(["cd " + shifuJobPath + " && echo Id >columns/meta.column.names"],shell=True)

    # shifu init
    subprocess.call(["cd " + shifuJobPath + " && bash shifu init > init.log"],shell=True)

    # calculate statistics
    subprocess.call(["cd " + shifuJobPath + " && bash shifu stats > stats.log"],shell=True)

    # normalize variables
    subprocess.call(["cd " + shifuJobPath + " && bash shifu norm " + "> norm.log"],shell=True)

    # variable selection
    subprocess.call(["cd " + shifuJobPath + " && bash shifu varsel > varsel.log"],shell=True)

    # model training
    subprocess.call(["cd " + shifuJobPath + " && bash shifu train > train.log"],shell=True)

    # calculate the metrics
    print (modelConfig["evals"][0])
    print (shifuJobPath)
    print (metricInfo)

    metric = evalMetrics(evalConfig=modelConfig["evals"][0],shifuJobPath=shifuJobPath,package="shifu",metricInfo=metricInfo)
    #change model name
    subprocess.call(["cd " + shifuJobPath + " && mv models/model0.nn ./modelLibrary/model_" + trialID + ".nn"],shell=True)
    subprocess.call(["cd " + shifuJobPath + " && mv ColumnConfig.json ./modelLibrary/ColumnConfig_" + trialID + ".json"],shell=True)
    print(metric)

if __name__ == "__main__":
    main()
