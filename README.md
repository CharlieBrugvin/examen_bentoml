# Examen BentoML

The exam consists of creating a web service that predicts the probability of admission to a university based on various features, such as GRE score, TOEFL score, university rating, etc. 

The model should be exposed as a secure REST API. To do so, we use BentoML and Docker.


The present repository contains the code to perform the following steps:
- Prepare the data
- Train a linear regression model
- Create a BentoML service
- Create a Docker image from the BentoML service
- Run the Docker image
- Run the tests to check the service

## Construct, run & test the Docker image


If you have the image archive `bento_image.tar`, you can unpack it with the command `docker load < bento_image.tar` and directly go to the step `II`.

### I - To construt the image `brugvin_admissions_prediction_service` from scratch, follow the steps below.

1. Install the requirements (tested with python 3.8)
```
pip install requirements.txt
```

<br>

2. Run the scripts to prepare data and the linear regression model:
```
python3 src/prepare_data.py
python3 src/train_model.py 
```
This should store a model named `admission_regression_model` into the BentoML model store.

<br>

3. Create the bento that encapsulates the model and the web service (implemented at `src/service.py`):
```
bentoml build
```
This should create a bento named `regression_model_service`.

<br>

4. Create a docker image from the bento:
```
bentoml containerize regression_model_service:latest --image-tag brugvin_admissions_prediction_service:1.0.0
```
This should create a docker image named `brugvin_admissions_prediction_service:1.0.0`.

### II - To run the image `brugvin_admissions_prediction_service` and the automatic tests, follow the steps below.

5. Run the docker image:
```
docker run --rm -p 3000:3000 brugvin_admissions_prediction_service:1.0.0
```

This should run the container. The swagger documentation is exposed at `http://localhost:3000`. 


6. Run the tests to check the service:
```
pytest test_api.py
```
6 tests should pass.