# Fashion Recommender System
This repository contains two scripts model.py and app.py which implement a fashion recommender system using the ResNet50 architecture from the TensorFlow library.

## model.py
model.py contains the code to extract features from the images using the ResNet50 architecture. The ResNet50 architecture is imported from the TensorFlow library and the layers are frozen so that they do not get trained during the feature extraction process. The images are loaded using the image module from the tensorflow.keras.preprocessing library and then they are preprocessed and passed through the ResNet50 model to obtain their feature vectors. These feature vectors are then normalized and stored in a list. Finally, the feature list and the filenames are stored in pickle files for later use by the app.py script.

## app.py
app.py contains the code to build a simple web app using the Streamlit library. The app allows a user to upload an image and then finds the five most similar images in the database and displays them to the user. The app uses the pickle files created by model.py to retrieve the feature list and filenames. The uploaded image is passed through the ResNet50 model to obtain its feature vector which is then compared to the feature list using the NearestNeighbors algorithm from the scikit-learn library to find the five most similar images. The images are displayed using the Streamlit library.

## Requirements
The following packages need to be installed in order to run the scripts:

tensorflow

streamlit

numpy

pickle

scikit-learn

pillow

## Usage
Run model.py to extract the features from the images and store them in pickle files.

Run app.py to start the web app.

Upload an image in the web app.

The app will display the five most similar images in the database.

## Note
Make sure to store the images in the images directory and the model.py script is run before running app.py to extract the features and store them in the pickle files.

