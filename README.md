# Extract Face Data

## Introduction
- The project aims to use deep learning methods to extract facial data from images for use in Illusion's games as character card facial data.
- Currently, only AI Shoujo and Honey Select2 are supported.

## Installation
- Firstly, you need to clone this project using the following command:
	- git clone https://github.com/ChasonJiang/Extract-Face-Data.git
- Then, you need to first install the following dependencies:
	- Python 3.8+ 
	- Pytorch
	- Numpy 
	- Opencv
	- Tensorboard
- You can use the following installation command:
	- pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
	- pip install numpy
	- pip install opencv-python
	- pip install tensorboard
- Recommend: It is recommended to use anaconda to install in a virtual environment

## Usage

- ### Extract
	- Step 1, Using extractor.py，Create an Extractor instance
		- extractor = Extractor()
	- Step 2, Extract the face data from image to json file
		- data=extractor.extract(<image_path>,<json_path>)
	- [Optional] Step 3, Print face data to the console
		- print(data)
	- [Optional]  You can find the initConfig() function in the Extractor class to modify the configuration of the Extractor.
- ### Train
	- Step 1, Download the dataset into project root dir.
    	- dataset [donwload link](https://pan.baidu.com/s/1l8Vtrrg93e9F_bRBgeooaw) code：3eo1
	- Step 2, Run train.py
	- [Optional]  You can find the initConfig() function in the Trainer class to modify the configuration of the Trainer.
- ### Evaluate
	- Step 1. Download the dataset into project root dir.
        - dataset [donwload link](https://pan.baidu.com/s/1l8Vtrrg93e9F_bRBgeooaw) code：3eo1
	- Step 2, Run evaluation.py
	- [Optional]  You can find the initConfig() function in the Evaluator class to modify the configuration of the Evaluator.

## Visualization
- Image
  
![avatar](./visualization/yuechan.png)


- in game
  
![avatar](./visualization/yuechan_in_game.png)

<details>
<summary>Face data</summary>
{
    "全脸宽度": 20,
    "脸上部前后位置": 35,
    "脸部上方和下方": 41,
    "下脸前后位置": 39,
    "脸下部宽度": 26,
    "下颚宽度": 21,
    "下巴上下位置1": 27,
    "下巴前后位置": 36,
    "下颚角度": 32,
    "下颚底部上下位置": 84,
    "下巴宽度": 41,
    "下巴上下位置2": 33,
    "下巴前后": 32,
    "脸颊下部上下位置": 53,
    "下颊前后": 33,
    "下颊宽度": 30,
    "脸颊上部上下位置": 52,
    "上颊前后": 52,
    "脸上部宽度": 35,
    "眼睛上下": 44,
    "眼位": 32,
    "眼睛前后": 25,
    "眼宽1": 44,
    "眼宽2": 31,
    "眼角z轴": 40,
    "视角y轴": 51,
    "左右眼位置1": 47,
    "左右眼位置2": 49,
    "眼角上下位置1": 54,
    "眼角上下位置2": 41,
    "眼皮形状1": 42,
    "眼皮形状2": 40,
    "整个鼻子上下位置": 40,
    "整个鼻子前后": 53,
    "鼻子整体角度X轴": 38,
    "鼻子的整个宽度": 46,
    "鼻梁高度": 32,
    "鼻梁宽度": 34,
    "鼻梁形状": 54,
    "鼻宽": 46,
    "上下鼻子": 45,
    "鼻子前后": 49,
    "机头角度X轴": 54,
    "机头角度Z轴": 45,
    "鼻子高度": 49,
    "鼻尖X轴": 44,
    "鼻尖大小": 35,
    "嘴上下": 65,
    "口宽": 44,
    "嘴唇宽度": 48,
    "嘴前后位置": 33,
    "上嘴唇形": 29,
    "下嘴唇形": 65,
    "嘴型嘴角": 37,
    "耳长": 39,
    "耳角Y轴": 59,
    "耳角Z轴": 53,
    "上耳形": 50,
    "耳下部形状": 45,
    "眉色": [
        27,
        30,
        22,
        80
    ],
    "唇色": [
        140,
        68,
        88,
        65
    ],
    "眼影颜色": [
        90,
        52,
        46,
        58
    ],
    "腮红颜色": [
        160,
        125,
        110,
        26
    ]
}
</details>




- Image
  
![avatar](./visualization/mlls.png)


- in game
  
![avatar](./visualization/mlls_in_game.png)

<details>
<summary>Face data</summary>
{
    "全脸宽度": 34,
    "脸上部前后位置": 22,
    "脸部上方和下方": 36,
    "下脸前后位置": 37,
    "脸下部宽度": 25,
    "下颚宽度": 25,
    "下巴上下位置1": 21,
    "下巴前后位置": 38,
    "下颚角度": 36,
    "下颚底部上下位置": 73,
    "下巴宽度": 24,
    "下巴上下位置2": 33,
    "下巴前后": 30,
    "脸颊下部上下位置": 50,
    "下颊前后": 28,
    "下颊宽度": 33,
    "脸颊上部上下位置": 40,
    "上颊前后": 45,
    "脸上部宽度": 38,
    "眼睛上下": 46,
    "眼位": 24,
    "眼睛前后": 17,
    "眼宽1": 37,
    "眼宽2": 35,
    "眼角z轴": 38,
    "视角y轴": 45,
    "左右眼位置1": 39,
    "左右眼位置2": 36,
    "眼角上下位置1": 54,
    "眼角上下位置2": 30,
    "眼皮形状1": 45,
    "眼皮形状2": 36,
    "整个鼻子上下位置": 33,
    "整个鼻子前后": 43,
    "鼻子整体角度X轴": 31,
    "鼻子的整个宽度": 38,
    "鼻梁高度": 25,
    "鼻梁宽度": 29,
    "鼻梁形状": 42,
    "鼻宽": 40,
    "上下鼻子": 39,
    "鼻子前后": 41,
    "机头角度X轴": 49,
    "机头角度Z轴": 34,
    "鼻子高度": 38,
    "鼻尖X轴": 40,
    "鼻尖大小": 27,
    "嘴上下": 60,
    "口宽": 33,
    "嘴唇宽度": 31,
    "嘴前后位置": 24,
    "上嘴唇形": 33,
    "下嘴唇形": 44,
    "嘴型嘴角": 38,
    "耳长": 37,
    "耳角Y轴": 47,
    "耳角Z轴": 44,
    "上耳形": 46,
    "耳下部形状": 37,
    "眉色": [
        46,
        52,
        34,
        81
    ],
    "唇色": [
        143,
        65,
        72,
        54
    ],
    "眼影颜色": [
        87,
        53,
        54,
        47
    ],
    "腮红颜色": [
        145,
        92,
        84,
        26
    ]
}
</details>



  
