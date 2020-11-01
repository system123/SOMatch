# SOMatch
A Framework for Deep Learning-based Sparse SAR-Optical Image Matching [[Paper](https://www.sciencedirect.com/science/article/pii/S0924271620302598?via%3Dihub)] 

## Building the Docker Image
`docker build -t somatch:latest .`

## Datasets and Dataloaders

## Training the Matching Network
`docker run -it --rm --runtime=nvidia -v <your dataset root>:/src/data/ -v <your results directory>:/src/results -e CUDA_VISIBLE_DEVICES=0 --ipc=host somatch:latest train.py --config config/so_match/experiments/backbone_asl1_wml.json`

## Creating the Goodness Network and ORN dataset

## Training the Goodness Networks

## Training the Outlier Reduction Network (ORN)

## Using the Trained Networks

# Using this work:
If you make use of this code, or generate any results using this code, please cite the corresponding paper:

> Hughes, L. H., Marcos, D., Lobry, S., Tuia, D., & Schmitt, M. (2020). A deep learning framework for matching of SAR and optical imagery. ISPRS Journal of Photogrammetry and Remote Sensing, 169, 166-179.

