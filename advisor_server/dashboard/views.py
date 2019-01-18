# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import json
import requests
import platform
import six
import os
import tempfile
import time

from django.contrib import messages
from django.shortcuts import redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import Http404
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from django.shortcuts import render_to_response
from django.conf import settings
from django import forms
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.decorators import login_required
import subprocess

from suggestion.models import Study
from suggestion.models import Trial
from suggestion.models import Champion


@login_required
def home(request):
  if request.user and request.user.is_authenticated:
    print(request.user.username)
  return render(request, "home.html")


def index(request):
  try:
    studies = [study.to_json() for study in Study.objects.all()]
  except Study.DoesNotExist:
    studies = []

  try:
    trials = [trial.to_json() for trial in Trial.objects.all()]
  except Study.DoesNotExist:
    trials = []

  packages = [f for f in os.listdir('../advisor_client/examples/')]

  try:
    champions = [champion.to_json() for champion in Champion.objects.all()]
  except Study.DoesNotExist:
    champions = []

  print("champions size = " + str(len(champions)));

  context = {
      "success": True,
      "studies": studies,
      "trials": trials,
      "packages": packages,
      "champions": champions,
      "platform": platform
  }
  return render(request, "index.html", context)


@csrf_exempt
def v1_studies(request):
  if request.method == "POST":
    name = request.POST.get("name", "")
    study_configuration = request.POST.get("study_configuration", "")
    algorithm = request.POST.get("algorithm", "RandomSearchAlgorithm")

    # Remove the charactors like \t and \"
    study_configuration_json = json.loads(study_configuration)
    data = {
        "name": name,
        "study_configuration": study_configuration_json,
        "algorithm": algorithm
    }

    url = "http://127.0.0.1:{}/suggestion/v1/studies".format(
        request.META.get("SERVER_PORT"))
    response = requests.post(url, json=data)
    messages.info(request, response.content)
    return redirect("index")
  else:
    response = {
        "error": True,
        "message": "{} method not allowed".format(request.method)
    }
    return JsonResponse(response, status=405)


@csrf_exempt
def v1_run_study(request):
  if request.method == 'POST':
    print(request.POST.get("name"))
    print(request.POST.get("study_configuration"))
    print(request.POST.get("trial-number"))
    print(request.POST.get("algorithm"))
    print(request.POST.get("ml-package"))

    ml_package = request.POST.get('ml-package')
    # find training script
    ml_path, ml_script = '', ''
    for f in os.listdir("../advisor_client/examples/" + ml_package):
      if f.endswith('.py'):
        ml_path = '../advisor_client/examples/' + ml_package
        ml_script = './' + f
        break

    print(request.POST)
    run_config = {
      "name": request.POST.get("name"),
      "algorithm": request.POST.get("algorithm"),
      "trialNumber": int(request.POST.get('trial-number') if request.POST.get('trial-number').strip() != '' else '10'),
      "path": ml_path,
      "command": ml_script,
      "search_space": json.loads(request.POST.get("study_configuration"))
    }

    new_file, filename = tempfile.mkstemp(suffix='.json')
    print(filename)
    os.write(new_file, json.dumps(run_config).encode())
    os.close(new_file)

    p = subprocess.Popen([os.getcwd() + "/../advisor_client/advisor_client/commandline/command.py", 
        "run", "-f", filename], stderr=subprocess.STDOUT, text=True)
    #print(p.returncode)
    #print(p.stdout)
    time.sleep(1)
    return redirect("index")
  else:
    response = {
      "error": True,
      "message": "{} method not allowed".format(request.method)
    }
    return JsonResponse(response, status=405)


@csrf_exempt
def v1_study(request, study_name):
  url = "http://127.0.0.1:{}/suggestion/v1/studies/{}".format(
      request.META.get("SERVER_PORT"), study_name)

  if request.method == "GET":
    response = requests.get(url)

    tirals_url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/trials".format(
        request.META.get("SERVER_PORT"), study_name)
    tirals_response = requests.get(tirals_url)

    if response.ok and tirals_response.ok:
      if six.PY2:
        study = json.loads(response.content.decode("utf-8"))["data"]
        trials = json.loads(tirals_response.content.decode("utf-8"))["data"]
      else:
        study = json.loads(response.text)["data"]
        trials = json.loads(tirals_response.text)["data"]
      min_id, min_val = trials[0]['id'], trials[0]['objective_value']
      for t in trials[1:]:
        if t['objective_value'] < min_val:
          min_id = t['id']
      context = {"success": True, "study": study, "trials": trials, "min_id": min_id}
      return render(request, "study_detail.html", context)
    else:
      response = {
          "error": True,
          "message": "Fail to request the url: {}".format(url)
      }
      return JsonResponse(response, status=405)
  elif request.method == "DELETE" or request.method == "POST":
    response = requests.delete(url)
    messages.info(request, response.content)
    return redirect("index")
  else:
    response = {
        "error": True,
        "message": "{} method not allowed".format(request.method)
    }
    return JsonResponse(response, status=405)

@csrf_exempt
def v1_champions(request):
  if request.method == "POST":
    data = {}
    url = "http://127.0.0.1:{}/suggestion/v1/champions".format(
      request.META.get("SERVER_PORT"))
    response = requests.get(url, json=data)
    messages.info(request, response.content)
    return redirect("index")
  else:
    response = {
      "error": True,
      "message": "{} method not allowed".format(request.method)
    }
    return JsonResponse(response, status=405)

@csrf_exempt
def v1_study_suggestions(request, study_name):
  if request.method == "POST":
    trials_number_string = request.POST.get("trials_number", "1")
    trials_number = int(trials_number_string)

    data = {"trials_number": trials_number}
    url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/suggestions".format(
        request.META.get("SERVER_PORT"), study_name)
    response = requests.post(url, json=data)
    messages.info(request, response.content)
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
  else:
    return JsonResponse({"error": "Unsupported http method"})


@csrf_exempt
def v1_trials(request):
  if request.method == "POST":
    study_name = request.POST.get("study_name", "")
    name = request.POST.get("name", "")

    data = {"name": name}

    url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/trials".format(
        request.META.get("SERVER_PORT"), study_name)
    response = requests.post(url, json=data)
    messages.info(request, response.content)
    return redirect("index")
  else:
    return JsonResponse({"error": "Unsupported http method"})


@csrf_exempt
def v1_trial(request, study_name, trial_id):
  url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/trials/{}".format(
      request.META.get("SERVER_PORT"), study_name, trial_id)

  if request.method == "GET":
    response = requests.get(url)

    tiral_metrics_url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/trials/{}/metrics".format(
        request.META.get("SERVER_PORT"), study_name, trial_id)
    tiral_metrics_response = requests.get(tiral_metrics_url)

    if response.ok and tiral_metrics_response.ok:
      if six.PY2:
        trial = json.loads(response.content.decode("utf-8"))["data"]
        trial_metrics = json.loads(
            tiral_metrics_response.content.decode("utf-8"))["data"]
      else:
        trial = json.loads(response.text)["data"]
        trial_metrics = json.loads(
          tiral_metrics_response.text)["data"]
      context = {
          "success": True,
          "trial": trial,
          "trial_metrics": trial_metrics
      }
      return render(request, "trial_detail.html", context)
    else:
      response = {
          "error": True,
          "message": "Fail to request the url: {}".format(url)
      }
      return JsonResponse(response, status=405)
  elif request.method == "DELETE":
    response = requests.delete(url)
    messages.info(request, response.content)
    return redirect("index")
  elif request.method == "PUT" or request.method == "POST":
    objective_value_string = request.POST.get("objective_value", "1.0")
    objective_value = float(objective_value_string)
    status = request.POST.get("status", "Completed")
    data = {"objective_value": objective_value, "status": status}
    response = requests.put(url, json=data)
    messages.info(request, response.content)

    if six.PY2:
      trial = json.loads(response.content.decode("utf-8"))["data"]
    else:
      trial = json.loads(response.text)["data"]
    context = {"success": True, "trial": trial, "trial_metrics": []}
    return render(request, "trial_detail.html", context)
  else:
    response = {
        "error": True,
        "message": "{} method not allowed".format(request.method)
    }
    return JsonResponse(response, status=405)


@csrf_exempt
def v1_study_trial_metrics(request, study_name, trial_id):
  if request.method == "POST":
    training_step_string = request.POST.get("training_step", "1")
    training_step = int(training_step_string)
    objective_value_string = request.POST.get("objective_value", "1.0")
    objective_value = float(objective_value_string)

    data = {"training_step": training_step, "objective_value": objective_value}
    url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/trials/{}/metrics".format(
        request.META.get("SERVER_PORT"), study_name, trial_id)
    response = requests.post(url, json=data)
    messages.info(request, response.content)
    return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
  else:
    return JsonResponse({"error": "Unsupported http method"})


@csrf_exempt
def v1_study_trial_metric(request, study_name, trial_id, metric_id):
  url = "http://127.0.0.1:{}/suggestion/v1/studies/{}/trials/{}/metrics/{}".format(
      request.META.get("SERVER_PORT"), study_name, trial_id, metric_id)

  if request.method == "GET":
    response = requests.get(url)

    if response.ok:
      if six.PY2:
        trial_metric = json.loads(response.content.decode("utf-8"))["data"]
      else:
        trial_metric = json.loads(response.text)["data"]
      context = {"success": True, "trial_metric": trial_metric}
      # TODO: Add the detail page of trial metric
      return render(request, "trial_detail.html", context)
    else:
      response = {
          "error": True,
          "message": "Fail to request the url: {}".format(url)
      }
      return JsonResponse(response, status=405)
  elif request.method == "DELETE" or request.method == "POST":
    response = requests.delete(url)
    messages.info(request, response.content)
    return redirect("index")
  else:
    response = {
        "error": True,
        "message": "{} method not allowed".format(request.method)
    }
    return JsonResponse(response, status=405)
