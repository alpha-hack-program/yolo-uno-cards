# Generated by KFP.
# Updated manually to include shared directory

ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ARG COMPONENT_NAME

WORKDIR /usr/local/src/kfp/components

COPY ${COMPONENT_NAME}/src/runtime-requirements.txt runtime-requirements.txt

RUN pip install --no-cache-dir -r runtime-requirements.txt
RUN pip install --no-cache-dir kfp==2.8.0

COPY shared ./shared
COPY ${COMPONENT_NAME}/src/ .
