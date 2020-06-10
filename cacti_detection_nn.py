#imports
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
import csv
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pylab
import skimage
import skimage.transform
import random

#import data
REBUILD_DATA = False
print("importing data")

class import_images():
    """
    Class to import cactus images for my computer file system
    """

    def __init__(self, Im_Size = (100, 100), train_pth=None):
        self.Im_Size = Im_Size
        self.train_pth = train_pth #directory of training images folder
        self.training_data = []
        self.corrupted_data = 0

    def make_training_data(self):
        for im in tqdm(os.listdir(train_pth)):
            try:
                path = os.path.join(train_pth, im)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, self.Im_Size)
                img_array = np.array(img)
                self.training_data.append((img_array, im))

            except:
                self.corrupted_data += 1
                print(self.corrupted_data)
                pass

        np.random.shuffle(self.training_data)
        np.save('cacti_cp.npy',self.training_data)

        return self.training_data, self.corrupted_data

if REBUILD_DATA:
    Im_Size = (32,32)
    train_pth = 'aerial-cactus-identification/train'

    cactus_data = import_images(Im_Size, train_pth)
    training_data, corrupted_data = cactus_data.make_training_data()

training_data = np.load('cacti_cp.npy',allow_pickle = True)

print("data imported")

#Importing labels from CSV file
print("importing labels")

f = open('/Users/rhuck/Documents/CompProjects/Research_NN_Practice/Deep-Learning-Projects/Cacti_Detection/Cactus-Detection-Neural-Network/train_cp.csv')
csv_f = csv.reader(f)
labels = dict(csv_f)
del labels['id']
# print(labels)
print("labels imported")


#Augmentation functions:

def horizontal_flip(img):
    return np.fliplr(img)

def vertical_flip(img):
    return np.flipud(img)

def rotation_clockwise(img):
    angle = random.randint(0, 180)
    return skimage.transform.rotate(img, angle = -angle)

def rotation_counterclockwise(img):
    angle = random.randint(0, 180)
    return skimage.transform.rotate(img, angle = angle)

def wrap_translation(img):
    ##Problem: distorts image, so might not be great to use
    transform = skimage.transform.AffineTransform(translation=(-200, 0))
    wrap_img = skimage.transform.warp(img, transform, mode = "wrap")
    return wrap_img

def noise(img):
    return skimage.util.random_noise(img)

def blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

transformations = {
    'horizontal_flip' : horizontal_flip,
    'vertical_flip' : vertical_flip,
    'rotation_clockwise' : rotation_clockwise,
    'rotation_counterclockwise' : rotation_counterclockwise,
    'wrap_translation' : wrap_translation,
    'noise' : noise,
    'blur' : blur
} #dict to map str to function names

transformations_list = [horizontal_flip, vertical_flip,
                        rotation_clockwise, rotation_counterclockwise,
                        wrap_translation, noise, blur]
#list of transformations


#Generate Augmented Data

AUGMENT_IMAGES = True
print("augmenting data")

if AUGMENT_IMAGES:
    #augment images - horizontal & vertical flips
    augmented_images = []
    for img, file in training_data:
        horiz_img = horizontal_flip(img)
        augmented_images.append((horiz_img, file))


        vert_img = vertical_flip(img)
        augmented_images.append((vert_img, file))


    augmented_data = np.array(augmented_images)

    #add augmented images to training_data
    full_data = np.concatenate((training_data, augmented_data))
    np.random.shuffle(full_data)
    np.save('cacti_augmented_cp.npy', full_data)
    # print(full_data)

full_data = np.load('cacti_augmented_cp.npy',allow_pickle = True)
print("data augmented")


#format training data labels by same index
print("formatting data")
data = []
data_labels = []
for image, file in full_data:
    is_cactus = int(labels[file])
    data.append(image)
    data_labels.append(is_cactus)

# print(data[:3])
# print(data_labels[:3])


#process data --> only train/dev (no test set)
#85/15 train/dev
print("building train/dev sets")
data_len = len(data)
train_len = data_len * 0.85
dev_len = data_len * 0.15

#perfect percentage values, so thats nice!
print("all data amount: ", data_len)
print("train data amount: ", train_len)
print("dev amount: ", dev_len)

train_data = data[:int(train_len)]
train_labels = data_labels[:int(train_len)]
# print(train_data[:3], train_labels[:3])
print(len(train_data), len(train_labels))

dev_data = data[int(train_len):]
dev_labels = data_labels[int(train_len):]
# print(dev_data[:3], dev_labels[:3])
print(len(dev_data), len(dev_labels))


#Helper Functions

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

#Define CNN

class CNN(nn.Module):
    def __init__(self, enter_channels, in_features, exit_classes, activation = 'relu'):
        super(CNN, self).__init__()

        self.enter_channels, self.exit_classes = enter_channels, exit_classes
        self.in_features, self.activation = in_features, activation

        #convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(enter_channels, 32, 5),
            activation_func(self.activation),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 5),
            activation_func(self.activation),
            nn.MaxPool2d(2, 2)
        )

        #fully connected layers
        self.fully_connected = nn.Sequential(
            nn.Linear(self.in_features, 512),
            activation_func(self.activation),

            nn.Linear(512, 256),
            activation_func(self.activation),

            nn.Linear(256 , exit_classes)
        )

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0), -1)

        x = self.fully_connected(x)
        return x

#Hyperparameters:
enter_channels, in_features, exit_classes, activation = 3, 64 * 5 * 5, 2, 'relu'

net = CNN(enter_channels, in_features, exit_classes, activation)


#Loss Function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001)

#Train Data
EPOCHS = 50
BATCH_SIZE = 35

dev_total = len(dev_data)

for epoch in range(EPOCHS):
    net.train(True)
    print("Train")
    for i in range(len(train_data) // BATCH_SIZE):
        data = train_data[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
        inputs = torch.tensor(data, dtype = torch.float)
        inputs /= 255.0
        inputs = inputs.reshape(-1, 3, 32, 32)

        labels = train_labels[i * BATCH_SIZE : (i+1) * BATCH_SIZE]
        labels = torch.tensor(labels, dtype = torch.long)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / BATCH_SIZE
        print(i, "loss ", loss.item(), "acc ", accuracy)#, outputs, labels)

    #epoch test on dev test
    net.train(False)
    print("Test")
    correct, score = 0, 0.0
    with torch.autograd.no_grad():
        for index, dev_input in enumerate(dev_data, 0):

            dev_input = torch.tensor(dev_input, dtype = torch.float)
            dev_input /= 255.0
            dev_input = dev_input.reshape(-1, 3, 32, 32)

            y = dev_labels[index]
            y = torch.tensor([y], dtype = torch.long)

            output = net(dev_input)

            score += float(criterion(output, y))

            _, predicted = torch.max(output.data, 1)
            correct += (predicted == y).sum().item()

        score /= dev_total
        accuracy = correct / dev_total
        print("\n", "\n", "\n", f'DEV TEST {epoch}')
        print("score: ", score)
        print("correct", correct)
        print("accuracy: ", accuracy)
        print("length: ", dev_total)
        print("\n", "\n", "\n")

    #Save training model
    PATH = f'./cactus_detector_cnn{epoch}.pth'
    torch.save(net.state_dict(), PATH)


#Plotting Functions
def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments:
    data     -- a list of dictionaries, each of which will be plotted
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for i in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

#Plot model Data

#Hyperparameters:
NUM_MODELS = 20
# **Modify PATH file** to get the correct models

#Needed Variables: train_data, train_labels, dev_data, dev_labels, test_data, test_labels

train_acc = {}
dev_acc = {}
test_acc = {}

train_scores = {}
dev_scores = {}
test_scores = {}

test_net = CNN(enter_channels, in_features, exit_classes, activation)

for epoch in range(NUM_MODELS):
    print(epoch)

    #load correct model
    PATH = f'./cactus_detector_cnn{epoch}.pth'
    test_net.load_state_dict(torch.load(PATH))
    test_net.eval()

    #Train set acc
    print("train set")
    train_correct, train_score = 0, 0.0
    train_total = len(train_data)
    with torch.autograd.no_grad():


        for index, train_input in enumerate(train_data, 0):

            train_input = torch.tensor(train_input, dtype = torch.float)
            train_input /= 255.0
            train_input = train_input.reshape(-1, 3, 32, 32)

            train_y = train_labels[index]
            train_y = torch.tensor([train_y], dtype = torch.long)

            train_output = test_net(train_input)

            train_score += float(criterion(train_output, train_y))

            _, predicted = torch.max(train_output.data, 1)
            train_correct += (predicted == train_y).sum().item()

        train_score /= train_total
        train_accuracy = train_correct / train_total
        print(train_score, train_accuracy)

        train_scores[epoch] = train_score
        train_acc[epoch] = train_accuracy


    #Dev set acc
    print("dev set")
    dev_correct, dev_score = 0, 0.0
    dev_total = len(dev_data)
    with torch.autograd.no_grad():


        for index, dev_input in enumerate(dev_data, 0):

            dev_input = torch.tensor(dev_input, dtype = torch.float)
            dev_input /= 255.0
            dev_input = dev_input.reshape(-1, 3, 32, 32)

            dev_y = dev_labels[index]
            dev_y = torch.tensor([dev_y], dtype = torch.long)

            dev_output = test_net(dev_input)

            dev_score += float(criterion(dev_output, dev_y))

            _, predicted = torch.max(dev_output.data, 1)
            dev_correct += (predicted == dev_y).sum().item()

        dev_score /= dev_total
        dev_accuracy = dev_correct / dev_total
        print(dev_score, dev_accuracy)

        dev_scores[epoch] = dev_score
        dev_acc[epoch] = dev_accuracy

print("train acc: ", train_acc)
print("train scores: ", train_scores)
print("dev acc: ", dev_acc)
print("dev scores: ", dev_scores)

#Plot Model Data

#Hyperparameters:
filename = Cacti_Detection_Results.png #"Name_of_File" #--give name if want to save file


plot_dict_list = [train_acc, train_scores, dev_acc, dev_scores]
plot_label = ["Train Set Accuracy", "Train Set Scores", "Dev Set Accuracy", "Dev Set Scores"]
plot_lines(plot_dict_list,"Accuracy of Models over Epochs","Epochs", "Accuracy/Score Value", plot_label, filename)
plt.show()
