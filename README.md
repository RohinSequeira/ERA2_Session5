# ERA2_Session5: Pytorch 101

Welcome to an introduction to Pytorch. 

Let's look at a sample code for training the MNIST data for identifying digits.

Our code is split into three files:


## 1. [S5.ipynb](S5.ipynb)

This is the main notebook where the flow of the code is defined.

## 2. [model.py](model.py)

Here you can see the structure of the Neural Network used for training our model.

## 3. [utils.py](utils.py)

All supplementary functionalities like data download, visualizations etc are persent in this file.

## Running the code

To run the code, you can simply clone the repo and start with [S5.ipynb](S5.ipynb). If you are using Google Colab, the way I have, make sure that the main notebook and supporting codes provided by [model.py](model.py) and [utils.py](utils.py) are saved onto your Google Drive and follow the two steps:

### 1) Mount your Google Drive:

```
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### 2) Change directory to where the files are located:

```
import os
os.chdir('/content/drive/MyDrive/ERA_V2/')
print("Current Working Directory:", os.getcwd())
```

Now you are good to proceed to run the code. Happy Learning!