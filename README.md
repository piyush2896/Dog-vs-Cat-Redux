# Dog vs Cat Redux Code
This repository contains my code for Dog vs Cat Redux challenge

## Directory Structure
Project
|-- datasets
|   |-- test
|   |-- test_set
|   |-- train
|   `-- train_set
|-- models
|-- submissions
|-- datalab.py
|-- dataset_clusterer.py
|-- make_file.py
|-- model.py
|-- predict.py
`-- train.py

## Usage
Run ```python dataset_clusterer.py``` to make batches of train data and test data and 
save them in ```./datasets/train_set``` and ```./datasets/train_set``` respectively.

Run ```python train.py``` to train the model and save it to ```./models/```

Run ```python predict.py``` to make probability predictions and save the output to ```./submissions/sub_1.csv```