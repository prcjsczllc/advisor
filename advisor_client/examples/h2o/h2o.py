#!/usr/bin/env python

import argparse
import subprocess
import json
import os
import h2o
from from h2o.estimators.glm import H2OGeneralizedLinearEstimator

parser = argparse.ArgumentParser()
parser.add_argument("-LearningRate", type=float, default=0.1)
parser.add_argument("-NumHiddenNodes", type=int, default=10)
parser.add_argument("-ActivationFunc", type=str, default="Tanh")
args = parser.parse_args()


def main():

    # start h2o cluster
    h2o.init()

    # read data
    data = h2o.import_file(path = "")
    data[1] = data[1].asfactor()

    # Compute or learning
    m = H2OGeneralizedLinearEstimator(family="binomial")
    m.train(x = fr.names[2:],y="CAPSULE",training_frame=data)
    subprocess.call(["cd " + shifuPath + " && bash shifu train > train.log"],shell=True)

    # Output the metrics
    logFileDir =  shifuPath + "/train.log"
    lines = [line.rsplit(' Validation Error: ')[1][:8] for line in open(logFileDir) if "Epoch #" in line and "Train Error:" in line and "Validation Error:" in line]
    validationError = min(lines)

    print(validationError)

if __name__ == "__main__":
    main()
