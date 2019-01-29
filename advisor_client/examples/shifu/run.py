#!/usr/bin/env python

import argparse
import subprocess
import json
import os
import sys
utilPath = "/risk_crm11/licliu/advisor/advisor_client/examples/"
sys.path.append(utilPath)
from utils.utils import readModelConfigStructure,trainEvalSplit,evalMetrics,setParamsHadoop

args = setParamsHadoop().parse_args()

def main():

    jobArgsList = ["trialID","studyName","metricInfo","useStats","useNormalizedData","useVariable","package","customPaths"]

    # get path
    currentPath = os.getcwd()
    if not os.path.isdir(currentPath + "/modelLibrary"):
        subprocess.call(["mkdir " + currentPath + "/modelLibrary"],shell=True)

    # Change model config
    with open(utilPath + "/utils/shifuModelConfigStructure.json","r+") as f:
        modelConfigStructure = json.loads(f.read())

    with open(currentPath + "/ModelConfig.json","r+") as modelConfigFile:
        modelConfig = json.loads(modelConfigFile.read())
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

    with open(currentPath + "/ModelConfig.json", "wt") as file:
        json.dump(modelConfig,file,indent=2,sort_keys=True)

    # calculate statistics
    StatsColumnConfigName="statsColumnConfig"
    if vars(args)["useStats"] != "true":
        for k,v in modelConfig["stats"]:
            StatsColumnConfigName += "_" + str(v)
        StatsColumnConfigName += ".json"
        if os.path.exists('modelLibrary/{0}'.format(StatsColumnConfigName)):
            subprocess.call(["cp modelLibrary/{0} ColumnConfig.json".format(StatsColumnConfigName)],shell=True)
        else:
            subprocess.call(["shifu stats > stats.log"],shell=True)
            subprocess.call(["cp ColumnConfig.json modelLibrary/{0}".format(StatsColumnConfigName)],shell=True)

    # normalize variables
    try:
        from subprocess import DEVNULL 
    except ImportError:
        DEVNULL = open(os.devnull, 'wb')

    normarlizedDataPath,normarlizedDataPathFull = [modelConfig["basic"]["customPaths"]["hdfsModelSetPath"] + "/" + modelConfig["basic"]["name"] + "/tmp/NormalizedData"] * 2 
    if modelConfig["basic"]["runMode"] <> "LOCAL":
        if vars(args)["useNormalizedData"] != "true":   
            for k,v in modelConfig["normalize"].items():
                normarlizedDataPathFull += "_" + str(v)
            if subprocess.call(["hadoop fs -ls " + normarlizedDataPathFull],shell=True,stderr=DEVNULL) != 0:                                                                                                                 
                subprocess.call(["shifu norm > norm.log"],shell=True)
                subprocess.call(["hadoop fs -cp {0} {1}".format(normarlizedDataPath,normarlizedDataPathFull)],shell=True) 
            modelConfig["train"]["customPaths"] =   {"normalizedDataPath":normarlizedDataPathFull}
            with open(currentPath + "/ModelConfig.json", "wt") as file:
                json.dump(modelConfig,file,indent=2,sort_keys=True)
    else:
        subprocess.call(["shifu norm > norm.log"],shell=True)
        modelConfig["train"]["customPaths"] =  ""
        with open(currentPath + "/ModelConfig.json", "wt") as file:
            json.dump(modelConfig,file,indent=2,sort_keys=True)

    # variable selection
    VarColumnConfigName="varColumnConfig"
    if vars(args)["useVariable"] != "true":
        for k,v in modelConfig["varSelect"]:
            if k not in ("candidateColumnNameFile","forceSelectColumnNameFile","forceRemoveColumnNameFile"):
                varColumnConfigName += "_" + str(v)
        varColumnConfigName += ".json"
        if os.path.exists('modelLibrary/{0}'.format(varColumnConfigName)):
            subprocess.call(["cp modelLibrary/{0} ColumnConfig.json".format(varColumnConfigName)],shell=True)
        else:
            subprocess.call(["shifu varsel > varsel.log"],shell=True)
            subprocess.call(["cp ColumnConfig.json modelLibrary/{0}".format(varColumnConfigName)],shell=True)

    # model training
    subprocess.call(["shifu train > train.log"],shell=True)

    # calculate the metrics
    metric = evalMetrics(modelConfig=modelConfig,shifuJobPath=currentPath,package="shifu",metricInfo=vars(args)["metricInfo"])
    #change model name
    subprocess.call(["cp models/model0.nn modelLibrary/model_" + vars(args)["trialID"] + ".nn"],shell=True)
    subprocess.call(["cp ColumnConfig.json modelLibrary/ColumnConfig_" + vars(args)["trialID"] + ".json"],shell=True)
    print(metric)

if __name__ == "__main__":
    main()
