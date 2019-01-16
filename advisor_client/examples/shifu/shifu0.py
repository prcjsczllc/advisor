#!/usr/bin/env python

import argparse
import subprocess
import json
import os
from utils import readModelConfigStructure,trainEvalSplit,evalMetrics,setDefaultParams
import pandas as np

args = setDefaultParams()

def getJobInput():
    return inputDict


def main():

    # get path
    currentPath = os.getcwd()

    # create a shifu job
    shifuJobName = "shifuJob"
    subprocess.call(["cd " + currentPath + " && bash shifu new " + shifuJobName],shell=True)

    #shifu job path
    shifuJobPath = currentPath + "/" + shifuJobName

    # split data to train and test
    inputFilePath="/home/licliu/advisor/advisor_client/examples/data/givemesomecredit/cs-training.csv"
    if not os.path.isfile(shifuJobPath + '/trainData.csv') or not os.path.isfile(shifuJobPath + '/trainData.csv'):
        trainEvalSplit(inputFilePath, shifuJobPath, 0.2,",")

    # Change model config
    with open(shifuJobPath + "/ModelConfig.json","r+") as modelConfigFile:
        modelConfig = json.loads(modelConfigFile.read())
        modelConfigStructure = readModelConfigStructure(modelConfig)

        # change job info
        # train data
        modelConfig["dataSet"]["dataPath"] = shifuJobPath + "/trainData.csv"
        modelConfig["dataSet"]["dataDelimiter"] = ","
        modelConfig["dataSet"]["headerPath"] = ""
        modelConfig["dataSet"]["headerDelimiter"] = ","
        modelConfig["dataSet"]["targetColumnName"] = "SeriousDlqin2yrs"
        modelConfig["dataSet"]["negTags"] = ["0"]
        modelConfig["dataSet"]["posTags"] = ["1"]

        # eval data
        modelConfig["evals"][0]["dataSet"]["dataPath"] =  shifuJobPath + "/evalData.csv"
        modelConfig["evals"][0]["dataSet"]["dataDelimiter"] = ","
        modelConfig["evals"][0]["dataSet"]["headerPath"] = ""
        modelConfig["evals"][0]["dataSet"]["headerDelimiter"] = ","
        modelConfig["evals"][0]["dataSet"]["targetColumnName"] = "SeriousDlqin2yrs"
        modelConfig["evals"][0]["dataSet"]["negTags"] = ["0"]
        modelConfig["evals"][0]["dataSet"]["posTags"] = ["1"]

        # training setting
        modelConfig["train"]["fixInitInput"] = true
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

    with open(shifuJobPath + "/ModelConfig.json", "wt") as file:
        json.dump(modelConfig,file,indent=2,sort_keys=True)

    # variable selection
    subprocess.call(["cd " + shifuJobPath + " && bash shifu varsel > varsel.log"],shell=True)

    # model training
    subprocess.call(["cd " + shifuJobPath + " && bash shifu train > train.log"],shell=True)

    # calculate the metrics
    metric = evalMetrics(evalConfig=modelConfig["evals"][0],shifuJobPath=shifuJobPath,package="shifu",metricType="auc")
    print(metric)

if __name__ == "__main__":
    main()
