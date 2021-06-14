# This is a project based on iWildcam 2021 - FGVC8
The task of this project is to count the number of animals of each species present in a sequence of images.
Data is omitted in this repository due to size issues, and can be downloaded from [kaggle](https://www.kaggle.com/c/iwildcam2021-fgvc8/data).
Store the downloaded data in a folder named data in the base directory to successfully run the notebooks and py files in this repository.

## Adjust Centroid Tracker
Head to the `SimpleObjectTracking` directory and alter `centroid_tracker.py`

## Train EfficientNet
Use the `256-x-256-cropped-images.ipynb` to crop animals detected in train images, and check out `efficientnet-with-undersampling.ipynb`
for training EfficientNet on these cropped images. The model pretrained for 300 epochs is named `300_.pth` in the base directory.

## Create submission
Head to the `SimpleObjectTracking` directory and execute
```
python run.py
```
which will create a submission csv file. Submit this file to [kaggle submissions page](https://www.kaggle.com/c/iwildcam2021-fgvc8/submit)
to get the final score for the submission.
