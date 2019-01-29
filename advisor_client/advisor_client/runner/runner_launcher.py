import json
import yaml
import logging
import subprocess
#import coloredlogs
import six
import getpass
import os

from .abstract_runner import AbstractRunner
from .local_runner import LocalRunner

from advisor_client.client import AdvisorClient

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("simple_tensorflow_serving")
logger.setLevel(logging.DEBUG)

#coloredlogs.install(
#   level='DEBUG', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')


class RunnerLauncher():
  def __init__(self, run_file=None):

    self.run_config_dict = {}

    if run_file:
        with open(run_file, "r") as f:
          if run_file.endswith(".json"):
            self.run_config_dict = json.load(f)
          elif run_file.endswith(".yml") or run_file.endswith(".yaml"):
            self.run_config_dict = yaml.safe_load(f)
          else:
            logging.error("Unsupport config file format, use json or yaml")

          # logging.info("Run with config: {}".format(self.run_config_dict))

  def run(self):
    logging.info("Run with config: {}".format(self.run_config_dict))
    # add endpoint
    client = AdvisorClient()

    # TODO: move the logic into local runner
    runner = LocalRunner()
    if "runner" in self.run_config_dict:
      if self.run_config_dict["runner"] == "local_runner":
        runner = LocalRunner()
        logging.info("Run with local runner")

    #get username
    username = getpass.getuser()
    if six.PY2:
      study_name = str(self.run_config_dict["name"].encode("utf-8")) + "_" + username
    else:
      study_name = str(self.run_config_dict["name"]) + "_" + username
    study = client.get_or_create_study(study_name,
                                       self.run_config_dict["search_space"],
                                       self.run_config_dict["algorithm"])
    #check whether data and metricinfo exists
    # if "data" in self.run_config_dict.keys():
    #     dataInfo = self.run_config_dict["data"]
    # else:
    #     dataInfo={}

    logging.info("Create study: {}".format(study))
    logging.info("------------------------- Start Study -------------------------")
    for i in range(self.run_config_dict["trialNumber"]):

      logging.info("-------------------- Start Trial --------------------")

      # Get suggested trials
      trials = client.get_suggestions(study.name, 1)
      logging.info("Get trial: {}".format(trials[0]))
      # Run training
      # generate parameters
      for trial in trials:
        parameters_dict = json.loads(trials[0].parameter_values)
        parameter_string = ""

        # add search space parameters
        for k, v in parameters_dict.items():
          parameter_string += " -{}={}".format(k, v)

        # if len(dataInfo)>0:
        #   for key,value in dataInfo.items():
        #     if key == "train" or key=="evals":
        #       for key2,value2 in value.items():
        #         parameter_string += " -{}={}".format(key2, value2.encode("utf-8"))
        #     elif key == "negTags" or key =="posTags":
        #       parameter_string += " -{}={}".format(key, [value[0].encode("utf-8")])
        #     else:
        #       parameter_string += " -{}={}".format(key,value.encode("utf-8"))
        if self.run_config_dict["package"] == "shifu":
          for k,v in self.run_config_dict["preCalculatedResults"].items():
            parameter_string += " -{}={}".format(k,v)
          parameter_string += " -{}={}".format("metricInfo",self.run_config_dict["search_space"]["metricInfo"])

        command_string = "{} {} -studyName={} -trialID={}".format(
            "python /risk_crm11/licliu/advisor/advisor_client/examples/" + self.run_config_dict["package"]+ "/run.py",
            parameter_string,study.name,trial.id)

        logging.info("Run the command: {}".format(command_string))


        # Example: '0.0\n'
        # Example: 'Compute y = x * x - 3 * x + 2\nIput x is: 1.0\nOutput is: 0.0\n0.0\n'
        if six.PY2:
          command_output = subprocess.check_output(command_string, shell=True)
        else:
          command_output = subprocess.check_output(command_string, universal_newlines=True, shell=True)
        # TODO: Log the output in the directory
        #logging.info("Get output of command: {}".format(command_output))

        metric = float(command_output.split("\n")[-2].strip())
        # Complete the trial
        client.complete_trial_with_one_metric(trial, metric)
        logging.info("Update the trial with metrics: {}".format(metric))

      logging.info("--------------------- End Trial ---------------------")
    logging.info("------------------------- End Study -------------------------")
    is_done = client.is_study_done(study.name)
    best_trial = client.get_best_trial(study.name)
    logging.info("The study: {}".format(study))
    logging.info("Best trial is: {}".format(best_trial))
