# EEG-Based Control of an Industrial Robotic Arm

This repository contains the code and data for a project aimed at developing a system to control an industrial robotic arm using EEG motor imagery related signals. The project is being carried out in the IPU lab, KTH.

## Project Overview

The project involves pre-processing EEG data with traditional signal filtering techniques, and then feeding this treated signal to a neural network. The goal is to interpret the motor imagery related patterns in the EEG signals to control the movements of an industrial robotic arm.

## Repository Structure

The repository is organized into two main folders:

### `NN`
This folder contains the neural network architectures developed for this project:

- **Sig_Cv2d_Lstm.py** : 1 2D Convolution with a kern of [22,12,12], 2 LSTM layers outputing 16 values on the last step of the sequence, 3 FC layers interlocked by Drops classifying into 4 classes
- **Snap_Cv2d.py** : 1 2D Convolution with a kern of [22,40,12], 4 2D convolution with kerns of [1,40,12*(2*(i+1))], 3 FC layers interlocked by Drops classifying into 4 classes
- **Sig_Cv1d_Lstm.py** : not developped
- **Snap_Cv1d.py** : not developped


# `data_MI`
This folder contains the EEG data, data preprocessing scripts, data loader, and data description:

- **DatasetDescription.txt** describes the semantic and organisation of the datas
- **translator.py** is a set of dictionnaries that synthesize the needed informations from the datasetdescription
- **DataInRam.py** allows to decrypt all the datas contained in the files and returns it in the demaded way. Can also apply basic filtering.

## Getting Started

To get started with this project, clone the repository and install the necessary dependencies :


- pip3 install mne
- pip3 install torch torchvision torchaudio
- pip3 install pytorch-lightning



## Contact

If you have any questions or feedback, please contact me via email : guillaume.flinois@ens-paris-saclay.fr

