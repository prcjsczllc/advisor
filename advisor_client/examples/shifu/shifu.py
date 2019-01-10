#!/usr/bin/env python

import argparse
import subprocess
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("-LearningRate", type=float, default=0.1)
parser.add_argument("-NumHiddenNodes", type=float, default=10)
parser.add_argument("-ActivationFunc", type=str, default="Tanh")
args = parser.parse_args()


def main():
    # Read parameters
    # LearningRate = args.LearningRate
    # NumHiddenNodes = args.NumHiddenNodes
    # ActivationFunc = args.ActivationFunc

    # get path
    shifuPath = "/home/licliu/advisor/advisor_client/examples/shifu/autoMLTest"

    # Change model config
    with open(shifuPath + "/ModelConfig.json","r+") as modelConfigFile:
        modelConfig = json.loads(modelConfigFile.read())
        for arg, value in vars(args).items():
            modelConfig["train"]["params"][arg] = [value]

    with open(shifuPath + "/ModelConfig.json", "wt") as t:
        json.dump(modelConfig,t,indent=2,sort_keys=True)

    # Compute or learning
    subprocess.check_call(["cd " + shifuPath + "&& bash shifu train && > train.log"],shell=True)

    # Output the metrics
    logFileDir =  shifuPath + "/train.log"
    lines = [line.rsplit(' Validation Error: ')[1][:8] for line in open(logFileDir) if "Epoch #" in line and "Train Error:" in line and "Validation Error:" in line]
    validationError = min(lines)

    print(validationError)

if __name__ == "__main__":
    main()
