# Required libraries

# data
import csv
import numpy as np
import skimage.io as skio
from tifffile import imread
from skimage import data, segmentation, feature, future

# diretory
import os
import glob

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV as RSCV

# graph
import seaborn as sns
from matplotlib import colors


# Set Up
img_Nbands = 4
random_state = 0
model_name = 'random_forest_'

class_dict = dict([
    (0, 'building'),
    (1, 'cultivated_area'),
    (2, 'forest'),
    (3, 'non_observed'),
    (4, 'other_uses'),
    (5, 'pasture'),
    (6, 'savanna_formation'),
    (7, 'water'),
])

colour_dict = dict([
    (0, 'lightgray'),
    (1, 'darkorange'),
    (2, 'forestgreen'),
    (3, 'darkslateblue'),
    (4, 'saddlebrown'),
    (5, 'olive'),
    (6, 'yellowgreen'),
    (7, 'royalblue'),
])

cmap = colors.ListedColormap(colour_dict.values())

bands = [f"B{i}" for i in range(1, img_Nbands + 1)]
print(f"Bands: {bands}")


def normalize0to1(data):
    # Normalize
    max_img = np.max(data)
    min_img = np.min(data)
    normalized_img = (data - min_img) / (max_img - min_img)
    #print(normalized_img)
    return normalized_img


def new_features(image):
    return 0


def get_image_size(image):
    img = imread(image)
    return img.shape


def get_image_dtype(image):
    img = imread(image)
    return img.dtype.name


# Yield successive n-sized chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


# Load images
def load_patches(path):

    # Create the sequence of the images path sat[0]
    image_path = path + "images/**/*.tif"
    # Take full path
    image_names = sorted(glob.glob(image_path))
    # File sort
    print(len(image_names), 'Images to train')

    # Create the sequence of the mask path sat[1]
    label_path = path + "masks_reference/**/*.tif"
    # Take full path
    label_names = sorted(glob.glob(label_path))
    print(len(label_names), 'Masks to train')

    qdt_samples = len(image_names)

    name_file = []

    # Read images to numpy array
    dimension = [len(image_names)] + [dim for dim in get_image_size(image_names[0])]
    XImages_train = np.zeros(dimension, dtype='float32') #dtype=get_image_dtype(image_names[0])

    print('Reading train images...')
    for i, img in enumerate(image_names):
        name_file.append(os.path.basename(img).split('.tif')[0])
        XImages_train[i] = normalize0to1(imread(img))

    # Read mask images to numpy array
    dimension = [len(label_names)] + [dim for dim in get_image_size(label_names[0])]
    YImages_train = np.zeros(dimension, dtype=get_image_dtype(label_names[0]))

    print('Reading train masks...')
    for i, img in enumerate(label_names):
        YImages_train[i] = imread(img)

    print('Dimensions of training dataset:')
    print('Images 3 channels (RGB):', XImages_train.shape)
    print('Mask 1 channel:', YImages_train.shape)

    XImages_train = np.reshape(XImages_train, [-1, 4])
    YImages_train = np.reshape(YImages_train, [-1])

    print('After reshape to features:')
    print('Images to RGB features:', XImages_train.shape)
    print('Mask to features:', YImages_train.shape)

    XImages_train = XImages_train[YImages_train != -1, :]  # remove NoData
    YImages_train = YImages_train[YImages_train != -1]  # remove NoData

    return XImages_train, YImages_train, name_file, qdt_samples


# Getting training data
train_dataset = '/Users/mateus.miranda/INPE-CAP/MSc/ai4luc/data/cerradatav3_1_splitted/train/'
X_train, y_train, name_file_train, qdt_TrainSamples = load_patches(train_dataset)

# Getting test data
train_dataset = '/Users/mateus.miranda/INPE-CAP/MSc/ai4luc/data/cerradatav3_1_DA/test/'
X_test, y_test, name_file_test, qdt_TestSamples = load_patches(train_dataset)


# Training
rf_clf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=9,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=0.9,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=1,
    random_state=random_state,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=0.5
)
""" 
# Selecting the best parameters
print('Selecting best parameters...')
param_grid = {'n_estimators': np.arange(50, 100, 150),
              'max_features': np.arange(0.1, 1, 0.1),
              'max_depth': [3, 5, 7, 9],
              'max_samples': [0.3, 0.5, 0.8]}

rf_clf = RSCV(RandomForestClassifier(), param_grid, n_iter = 15).fit(X_train, y_train)

print("Best parameters: ", rf_clf.best_params_)
print("Best classification model: ", rf_clf.best_estimator_)

rf_model = rf_clf.best_estimator_
"""

print('Training...')
#rf_model.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# Prediction
path_prediction = '/Users/mateus.miranda/INPE-CAP/MSc/ai4luc/report/cap_acrp/rf_outputs/'
print('Mean accuracy of training:', rf_clf.score(X_train, y_train))

print('Predicting... ')

rf_predicted_arr = rf_clf.predict(X_test)
y_pred = rf_predicted_arr.reshape(qdt_TestSamples, 256, 256)
print(np.shape(y_pred))

print('Saving the predictions...')
for xt in range(len(y_pred)):
    # Saving
    skio.imsave(path_prediction + '/ypred_sDA_' + model_name + name_file_test[xt] + '.tif', y_pred[xt], check_contrast=False)


