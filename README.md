# Self-Supervised Monocular Depth Underwater

This repository contains the code for the paper "Self-Supervised Monocular Depth Underwater". The work is primarily based on DiffNet.

Paper URL: [Self-Supervised Monocular Depth Underwater](https://arxiv.org/abs/2210.03206)  
DiffNet Github: [DIFFNet](https://github.com/brandleyzhou/DIFFNet)

## Prerequisites
The following python packages are required to run the application. 

```
matplotlib==3.4.2
numpy==1.21.2
opencv-python==4.5.2.52
Pillow==8.4.0
scikit-image==0.18.3
scipy==1.7.1
tensorboard==2.7.0
tensorboardX==2.4
torch==1.10.1
torchvision==0.2.1
```

## Instructions
The model can be trained and evaluated on your local machine, or using Docker to ensure environment consistency.

1. **Training and Evaluation**

There are two shell scripts, `train.sh` and `evaluate.sh` for training and evaluation of the model. Ensure to replace the `<datapath>` and `<weightsFolder>` with your actual data path and weights folder respectively in all shell scripts. 

2. **Docker Setup**

You can use Docker to run the scripts. Docker ensures that the environment is consistent across different machines. Docker will automatically build the image if it's not present locally.

To train the model using Docker:

```sh
# Ensure you're in the project directory
./docker_train.sh
```
This will automatically execute the `train.sh` script inside a Docker container.

To evaluate the model using Docker:

```sh
# Ensure you're in the project directory
./docker_eval.sh
```
This will automatically execute the `evaluate.sh` script inside a Docker container.

## Feedback

If you encounter any issues or have questions, feel free to open an issue in this repository. Your feedback is highly appreciated!