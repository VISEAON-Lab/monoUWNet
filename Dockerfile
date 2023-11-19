FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel
RUN pip install tensorboardX==2.4 opencv-python==4.5.2.52 matplotlib==3.4.2 numpy==1.21.2 Pillow==8.4.0 scikit-image==0.18.3 scipy==1.7.1 tensorboard==2.7.0 torchvision==0.2.1  protobuf==3.20.0
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN pip install yacs
RUN apt-get install -y git

# Inference 
RUN pip install pykuwahara
RUN pip uninstall -y opencv-python
RUN pip install opencv-python[gstreamer]
