#!/usr/bin/env python

import argparse
import subprocess
import json
import os
import sys
currentPath = os.getcwd()
sys.path.append("/home/licliu/advisor/advisor_client/examples/")
from utils.utils import readModelConfigStructure,trainEvalSplit,evalMetrics,setParamsHadoop

args = setParamsHadoop().parse_args()

def main():

    jobArgsList = ["trialID","studyName","metricInfo","targetColumnName","negTags","posTags","categoricalColumnNames","metaColumnNames","dataDelimiter","dataHeader",\
    "trainDataPath","trainDataWeightColumnName","trainDataFilterExpressions","evalDataPath","evalDataWeightColumnName","evalDataFilterExpressions","performanceBucketNum","Package"]
    #get optimzation info
    metricInfo=vars(args)["metricInfo"]

    # get path
    currentPath = os.getcwd()

    # create a shifu job
    shifuJobName = vars(args)["studyName"]
    #shifu job path
    shifuJobPath = currentPath + "/" + shifuJobName
    print(shifuJobPath)
    subprocess.call(["cd " + currentPath + " && shifu new " + shifuJobName],shell=True)
    if not os.path.isdir( shifuJobPath + "/modelLibrary"):
        subprocess.call(["mkdir " + shifuJobPath + "/modelLibrary"],shell=True)

    # Change model config
    with open(shifuJobPath + "/ModelConfig.json","r+") as modelConfigFile:
        modelConfig = json.loads(modelConfigFile.read())
        modelConfigStructure = readModelConfigStructure(modelConfig)

        # change job info
        # train data
        modelConfig["dataSet"]["dataPath"] = vars(args)["trainDataPath"]
        modelConfig["dataSet"]["dataDelimiter"] = vars(args)["dataDelimiter"]
        modelConfig["dataSet"]["headerPath"] = vars(args)["dataHeader"]
        modelConfig["dataSet"]["headerDelimiter"] = vars(args)["dataDelimiter"]
        modelConfig["dataSet"]["targetColumnName"] = vars(args)["targetColumnName"]
        modelConfig["dataSet"]["negTags"] = [str(vars(args)["negTags"][1:-1])]
        modelConfig["dataSet"]["posTags"] = [str(vars(args)["posTags"][1:-1])]
        modelConfig["dataset"]["categoricalColumnNameFile"]= vars(args)["categoricalColumnNameFile"]
        modelConfig["dataset"]["metaColumnNameFile"]= vars(args)["metaColumnNameFile"]

        # eval data
        modelConfig["evals"][0]["dataSet"]["dataPath"] =  vars(args)["evalDataPath"]
        modelConfig["evals"][0]["dataSet"]["dataDelimiter"] = vars(args)["dataDelimiter"]
        modelConfig["evals"][0]["dataSet"]["headerPath"] =  vars(args)["dataHeader"]
        modelConfig["evals"][0]["dataSet"]["headerDelimiter"] = vars(args)["dataDelimiter"]
        modelConfig["evals"][0]["dataSet"]["targetColumnName"] = vars(args)["targetColumnName"]
        modelConfig["evals"][0]["dataSet"]["negTags"]= [str(vars(args)["negTags"][1:-1])]
        modelConfig["evals"][0]["dataSet"]["posTags"] = [str(vars(args)["posTags"][1:-1])]
        modelConfig["evals"][0]["performanceBucketNum"] = "100"

        # training setting
        # change optimizing target parameters

        # assign arguements
        for arg, value in vars(args).items():
            if arg in jobArgsList:
                continue
            # stats
            elif modelConfigStructure[arg] == "stats":
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

    metric = evalMetrics(modelConfig=modelConfig,shifuJobPath=shifuJobPath,package="shifu",metricInfo=metricInfo)
    #change model name
    subprocess.call(["cd " + shifuJobPath + " && mv models/model0.nn ./modelLibrary/model_" + vars(args)["trialID"] + ".nn"],shell=True)
    subprocess.call(["cd " + shifuJobPath + " && mv ColumnConfig.json ./modelLibrary/ColumnConfig_" + vars(args)["trialID"] + ".json"],shell=True)
    print(metric)

if __name__ == "__main__":
    main()
