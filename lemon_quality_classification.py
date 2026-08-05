import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import mahotas as mh

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay

import asyncio
import skillsnetwork

async def main():
    await skillsnetwork.prepare(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-GPXX0UEREN/hiroshima-lemon.zip",
        overwrite=True
    )

asyncio.run(main())
def get_data(folder, file):
    plt.rcParams["axes.grid"] = False
    data = []
    print(folder + '/' + file)
    ds = pd.read_csv(folder + '/' + file + '.csv') # Form a DataSet from a csv
    print(ds)
    i = 0
    r = 0
    for im, c in ds.values:
        if i==0 and r < 50:
            fig = plt.figure(figsize = (20,20)) # Display an image
        image = mh.imread(folder + '/' + file + '/' + im) #Download an image
        data.append([image, c]) # Append finaly DataSet
        if r < 50:
            plt.subplot(1, 5, i+1) # Create a table of images
            plt.imshow(image)
        i += 1
        if i==5:
            i = 0
        r += 1
    plt.show()

    return np.array(data, dtype=object)

d = "hiroshima-lemon"
f = "train_images"
train = get_data(d, f)

"""This step applies the data-loading function to the training folder and constructs the full training dataset. The resulting variable train contains all lemon images along with their quality labels. The successful display of images confirms that the data has been loaded correctly and that the dataset is usable for feature extraction and model training."""

import os

for file in os.listdir("hiroshima-lemon"):
    print(file)

"""This command lists the contents of the dataset directory to verify the presence of required folders such as training images, test images, and CSV annotation files. Confirming this structure ensures that subsequent processing steps will not fail due to missing files."""

c = pd.DataFrame(train[:, 1])
c.columns = ['Class']
sns.set_style('darkgrid')
sns.countplot(x="Class", data=c)

"""A count plot is show how many samples belong to each lemon quality class. This visualization reveals whether the dataset is balanced or imbalanced across classes. If one class contains significantly more samples than others, it may influence model learning and prediction bias. Understanding class distribution at this stage helps in interpreting later model performance."""

class_names = {0: 'excellent', 1: 'good', 2: 'processed products', 3: 'disqualified'}

"""Numeric class labels are mapped to meaningful textual names. This improves interpretability of plots, predictions, and confusion matrices by replacing numeric codes with understandable quality categories."""

# sklearn function
from sklearn.model_selection import train_test_split
im_train, im_val, c_train, c_val = train_test_split(train[:, 0], train[:, 1], test_size=0.3, shuffle=True)
c_train = c_train.astype('int')
c_val = c_val.astype('int')

"""The dataset is randomly divided into training (70%) and validation (30%) subsets. The training set is used to build the model, while the validation set is reserved for unbiased performance evaluation. This separation prevents overfitting and allows an objective assessment of model generalization."""

print("Train shape", im_train.shape) # Size of the training DataSet
print("Test shape", im_val.shape) # Size of the test DataSet

"""These outputs indicate the number of images allocated to training and validation. This confirms that data splitting has occurred correctly and that sufficient samples exist in both subsets."""

c = pd.DataFrame(c_train)
c.columns = ['Train Class']
c['Train Class Name'] = c['Train Class'].map(class_names)
sns.set_style('darkgrid')
sns.countplot(x="Train Class Name", data=c, order = ['excellent', 'good', 'processed products', 'disqualified'])

"""This plot shows how lemon quality classes are distributed specifically within the training data. A similar distribution to the full dataset indicates that the split preserved class proportions, which is important for reliable model training."""

c = pd.DataFrame(c_val)
c.columns = ['Train Class']
c['Validate Class Name'] = c['Train Class'].map(class_names)
sns.set_style('darkgrid')
sns.countplot(x="Validate Class Name", data=c, order = ['excellent', 'good', 'processed products', 'disqualified'])

"""This visualization checks whether validation data also maintains a balanced representation of each class. Balanced validation data ensures that evaluation results are fair and meaningful."""

print("Quality of lemon")
plt.rcParams["axes.grid"] = False
for c in class_names:
    ims = np.where(c_train == c)[0][:5]
    print(class_names[c])
    fig = plt.figure(figsize = (20,20))
    for i, im in enumerate(ims):
        plt.subplot(1, 5, i+1) #create a table of images
        plt.imshow(im_train[im])
        plt.title(class_names[c])
    plt.show()

"""Example images from each quality category are displayed. This helps visually verify that classes correspond to distinct visual characteristics such as surface smoothness, color consistency, and defects. It confirms that visual features can reasonably be used for classification."""

def create_features(im):
    features = []
    for image in im:
        im_grey = mh.colors.rgb2grey(image, dtype = np.uint8)
        features.append(mh.features.haralick(im_grey).ravel())
    features = np.array(features)
    return (features)

features_train = create_features(im_train)
features_val = create_features(im_val)

"""Each image is converted to grayscale and processed to extract Haralick texture features, which describe patterns of pixel intensity variations. These features quantify surface texture and structural properties of lemons, transforming raw images into numerical representations suitable for machine learning."""

print(features_train.shape)
print(features_val.shape)

"""This confirms that feature extraction was successful and consistent for both training and validation data."""

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000))])
clf.fit(features_train, c_train)
scores_train = clf.score(features_train, c_train)
scores_val = clf.score(features_val, c_val)
print('Training DataSet accuracy: {: .1%}'.format(scores_train), 'Test DataSet accuracy: {: .1%}'.format(scores_val))
ConfusionMatrixDisplay.from_estimator(clf, features_val, c_val)
plt.show()

"""The confusion matrix displays correct and incorrect predictions for each class. Diagonal cells represent correct classifications, while off-diagonal cells show misclassifications. This helps identify which lemon quality categories are most frequently confused."""

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    LogisticRegression(max_iter=1000),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
scores_train = []
scores_val = []
for name, clf in zip(names, classifiers):
    print("Fitting:", name)
    clf = Pipeline([('preproc', StandardScaler()), ('classifier', clf)])
    clf.fit(features_train, c_train)
    score_train = clf.score(features_train, c_train)
    score_val = clf.score(features_val, c_val)
    scores_train.append(score_train)
    scores_val.append(score_val)

res = pd.DataFrame(index = names)
res['scores_train'] = scores_train
res['scores_val'] = scores_val
res.columns = ['Test','Validate']
res.index.name = "Classifier accuracy"
pd.options.display.float_format = '{:,.2f}'.format
print(res)

x = np.arange(len(names))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, scores_train, width, label='Train')
rects2 = ax.bar(x + width/2, scores_val, width, label='Test')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy of classifiers')
ax.set_xticks(x)
plt.xticks(rotation = 90)
ax.set_xticklabels(names)
ax.legend()

fig.tight_layout()

plt.show()

"""The bar chart visually compares training and validation accuracies of each classifier, making it easy to identify strong and weak models."""

def get_test_data(folder, file):
    data = []
    print(folder + '/' + file)
    ds = pd.read_csv(folder + '/' + file + '.csv')
    for im in ds.values:
        image = mh.imread(folder + '/' + file + '/' + im[0])
        data.append(image)
    return np.array(data)

d = "hiroshima-lemon"
f = "test_images"
test = get_test_data(d, f)

features_test = create_features(test)

print(test.shape)
print(features_test.shape)

clf = Pipeline([('preproc', StandardScaler()), ('classifier', LogisticRegression(max_iter=1000))])
clf.fit(features_train, c_train)

c_test = clf.predict(features_test)
c_test

"""This indicates that each element in the array corresponds to the predicted quality category of a test image in the same order as the input data. The numeric values represent lemon quality classes (0 = Excellent, 1 = Good, 2 = Processed Products, 3 = Disqualified). This confirms that the model successfully produced predictions for every unseen test sample, converting extracted image features into meaningful quality classifications."""

print("Classification results")
for c in class_names:
    ims = np.where(c_test == c)[0][:5]
    print(class_names[c])
    fig = plt.figure(figsize = (20,20))
    for i, im in enumerate(ims):
        plt.subplot(1, 5, i+1) #create table of images
        plt.imshow(test[im])
        plt.title(class_names[c])
    plt.show()

"""For each predicted class, I displayed example test images. The visual appearance of these images aligned well with their predicted quality categories, supporting the reliability of the model’s predictions."""

def safe_test_result(folder, file, res):
    print(folder + '/' + file)
    ds = pd.read_csv(folder + '/' + file + '.csv')
    ds['Class'] = res
    print(ds)
    ds.to_csv('results.csv', index=False)
    return

d = "hiroshima-lemon"
f = "test_images"
safe_test_result(d, f, c_test)

"""The execution of safe_test_result(d, f, c_test) generated prediction results for 1651 test images, producing a table with two columns: image ID and predicted class label. Each image was successfully assigned to one of four lemon quality categories (0 = Excellent, 1 = Good, 2 = Processed Products, 3 = Disqualified). This confirms that the trained model was able to classify all unseen test images without errors. The results were also saved into a file named results.csv, which represents the final output of the lemon quality classification system."""