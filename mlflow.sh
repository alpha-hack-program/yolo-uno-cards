#!/bin/sh

. .env

mlflow server --backend-store-uri runs/mlflow