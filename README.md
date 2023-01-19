#  Image segmentation of zebrafish images

This project tests three neural network models made for image segmentation on a dataset of zebrafish images. Dataset is not included but can be downloaded using `download_dataset.py`. The trained models are not included as well as they are too heavy for github. 

The three models used in this project are referred to **binary model**, **cropped model**, and **multi-class model**. 

The **binary model** performs basic binary image segmentation. `run_binary_model.ipynb` is used to train, validate and evaluate such model. It is also used to study the performances of the model for different threshold value. This can also be done using `study_threshold.ipynb`. `predict_binary.ipynb` is used to generate segmentation masks using the binary model.

The **cropped model** also preforms binary segmentation but feeds images cropped round the head of the fish to the network. `run_cropped_model.ipynb` is used to 
train, validate and evaluate such model. It is also used to study the performances of the model for different threshold value. This can also be done using `study_threshold.ipynb`. `predict_cropped_model.ipynb` is used to generate segmentation masks using the cropped model.

Finally, the **multi-class model** performs basic multi-class image segmentation. `run_multiclass_model.ipynb` is used to train, validate, and evaluate the performances of the model globally. `study_threshold_multi.ipynb` can be used to study the performances per annotation for multiple threshold values. `predict_multiclass_model.ipynb` is used to generated segmentation masks using the binary model. One mask is created per annotation.

