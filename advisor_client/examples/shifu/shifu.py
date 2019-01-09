#!/usr/bin/env python

import argparse
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument("-LearningRate", type=float, default=0.1)
parser.add_argument("-NumHiddenNodes", type=float, default=10)
parser.add_argument("-ActivationFunc", type=str, default="tanh")
args = parser.parse_args()


def main():
  # Read parameters
  LearningRate = args.LearningRate
  NumHiddenNodes = args.NumHiddenNodes
  ActivationFunc = args.ActivationFunc

  # Compute or learning
  subprocess.check_call(["shifu train"])

  # Output the metrics
  print(y)


if __name__ == "__main__":
  main()
