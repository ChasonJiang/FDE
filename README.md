# Extract Face Data

## Introduction
- The project aims to use deep learning methods to extract facial data from images for use in Illusion's games as character card facial data.
- Currently, only AI Shoujo and Honey Select2 are supported.

## Installation
- To install this project, you need to first install the following dependencies:
	- Python 3.8+ 
	- Pytorch
	- Numpy 
	- Opencv
	- TensorboardX

- You can use the following installation command:

## Usage

- ### Extract
	- Step 1, Using extractor.py，Create an Extractor instance
		extractor = Extractor()
	- Step 2, Extract the face data from image to json file
		data=extractor.extract(<image_path>,<json_path>)
	- [Optional] Step 3, Print face data to the console
		print(data)
	- [Optional]  You can find the initConfig() function in the Extractor class to modify the configuration of the Extractor.
- ### Train
	- Step 1, Download the dataset into project root dir.
	- Step 2, Run train.py
	- [Optional]  You can find the initConfig() function in the Trainer class to modify the configuration of the Extractor.
- ### Evaluate
	- Step 1. Download the dataset into project root dir.
	- Step 2, Run evaluation.py
	- [Optional]  You can find the initConfig() function in the Evaluator class to modify the configuration of the Extractor.
