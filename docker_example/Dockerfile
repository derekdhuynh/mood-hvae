FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04
#FROM nvidia/cuda:11.3.1-base-ubuntu20.04

FROM python:3.8.10
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

#FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

# copy files
ADD scripts /workspace/
ADD models /workspace/models
#ENV PYTHONPATH /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred

RUN pip install nibabel
RUN pip install torchvision
RUN pip install matplotlib
RUN pip install seaborn
RUN pip install torchvision
RUN pip install tqdm
RUN pip install rich
RUN pip install black
