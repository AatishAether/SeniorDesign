# import os
import csv
import numpy as np
# import pandas as pd
# import torch
# import cv2
from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt
# import random

#for modded ASLDataset Dataloader
from PIL import Image

#The model itself
#from model

#from dataset.py from tutorial
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_set = r'.\tP\alph_train'
test_set = r'.\tP\alph_test'

# def build_csv(directory_string, output_csv_name):
#     """Builds a csv file for pytorch training from a directory of folders of images.
#     Install csv module if not already installed.
#     Args:
#     directory_string: string of directory path, e.g. r'.\data\train'https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
#     output_csv_name: string of output csv file name, e.g. 'train.csv'
#     Returns:
#     csv file with file names, file paths, class names and class indices
#     """
#     import csv
#     directory = directory_string
#     class_lst = os.listdir(directory) #returns a LIST containing the names of the entries (folder names in this case) in the directory.
#     class_lst.sort() #IMPORTANT
#     with open(output_csv_name, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile, delimiter=',')
#         writer.writerow(['file_name', 'file_path', 'class_name', 'class_index']) #create column names
#         for class_name in class_lst:
#             class_path = os.path.join(directory, class_name) #concatenates various path components with exactly one directory separator (‘/’) except the last path component.
#             file_list = os.listdir(class_path) #get list of files in class folder
#             for file_name in file_list:
#                 file_path = os.path.join(directory, class_name, file_name) #concatenate class folder dir, class name and file name
#                 writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) #write the file path and class name to the csv file
#     return
#
# build_csv(train_set, 'train.csv')
# build_csv(test_set, 'test.csv')
# train_df = pd.read_csv('train.csv')
# test_df = pd.read_csv('test.csv')
#
# class_zip = zip(train_df['class_index'], train_df['class_name'])
# my_list = []
# for index, name in class_zip:
#   tup = tuple((index, name))
#   my_list.append(tup)
# unique_list = list(set(my_list))
# print('Training:')
# # print(sorted(unique_list))
# print()
#
# class_zip = zip(test_df['class_index'], test_df['class_name'])
# my_list = []
# for index, name in class_zip:
#   tup = tuple((index, name))
#   my_list.append(tup)
# unique_list = list(set(my_list))
# print('Testing:')
# # print(sorted(unique_list))
#
# class_names = list(train_df['class_name'].unique())

# class ASLDataset(Dataset): # inheriting from Dataset class
#     def __init__(self, csv_file, root_dir="", transform=None):
#         self.annotation_df = pd.read_csv(csv_file)
#         self.root_dir = root_dir # root directory of images, leave "" if using the image path column in the __getitem__ method
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.annotation_df) # return length (numer of rows) of the dataframe
#
#     def __getitem__(self, idx):
#         print("reached")
#         image_path = os.path.join(self.root_dir, self.annotation_df.iloc[idx, 1]) #use image path column (index = 1) in csv file
#
#         image = cv2.imread(image_path) # read image by cv2
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB for matplotlib
#         class_name = self.annotation_df.iloc[idx, 2] # use class name column (index = 2) in csv file
#         class_index = self.annotation_df.iloc[idx, 3] # use class index column (index = 3) in csv file
#         if self.transform:
#             image = self.transform(image)
#         return image, class_name, class_index

# #The following is an attempt at a modified attribute loader from a tutorial
# class AttributesDataset():
#     def __init__(self, annotation_path):
#         letter_labels = []
#         # gender_labels = []
#         # article_labels = []
#
#         with open(annotation_path) as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 letter_labels.append(row['class_name'])
#                 # gender_labels.append(row['gender'])
#                 # article_labels.append(row['articleType'])
#
#         self.letter_labels = np.unique(letter_labels)
#         # self.gender_labels = np.unique(gender_labels)
#         # self.article_labels = np.unique(article_labels)
#
#         self.num_letters = len(self.letter_labels)
#         # self.num_genders = len(self.gender_labels)
#         # self.num_articles = len(self.article_labels)
#
#         self.letter_id_to_name = dict(zip(range(len(self.letter_labels)), self.letter_labels))
#         self.letter_name_to_id = dict(zip(self.letter_labels, range(len(self.letter_labels))))
#
#         # self.gender_id_to_name = dict(zip(range(len(self.gender_labels)), self.gender_labels))
#         # self.gender_name_to_id = dict(zip(self.gender_labels, range(len(self.gender_labels))))
#         #
#         # self.article_id_to_name = dict(zip(range(len(self.article_labels)), self.article_labels))
#         # self.article_name_to_id = dict(zip(self.article_labels, range(len(self.article_labels))))
#
# #The above dataloader didn't work as expected. Here is one modified from a tutorial:
# class ASLDataset(Dataset):
#     def __init__(self, annotation_path, attributes, transform=None):
#         super().__init__()
#
#         self.transform = transform
#         self.attr = attributes
#
#         # initialize the arrays to store the ground truth labels and paths to the images
#         self.data = []
#         self.letter_labels = [] #color
#         # self.gender_labels = []
#         # self.article_labels = []
#
#         # read the annotations from the CSV file
#         with open(annotation_path) as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 self.data.append(row['file_path']) #'image_path'
#                 self.letter_labels.append(self.attr.letter_name_to_id[row['class_name']]) #color  'baseColour'
#                 # self.gender_labels.append(self.attr.gender_name_to_id[row['gender']])
#                 # self.article_labels.append(self.attr.article_name_to_id[row['articleType']])
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # take the data sample by its index
#         img_path = self.data[idx]
#
#         # read image
#         img = Image.open(img_path)
#
#         # apply the image augmentations if needed
#         if self.transform:
#             img = self.transform(img)
#
#         # return the image and all the associated labels
#         dict_data = {
#             'img': img,
#             'labels': {
#                 'letter_labels': self.letter_labels[idx] #color,
#                 # 'gender_labels': self.gender_labels[idx],
#                 # 'article_labels': self.article_labels[idx]
#             }
#         }
#         return dict_data
#
# #test dataset class without transformation:
# #train_dataset_untransformed = ASLDataset(csv_file='train.csv', root_dir="", transform=None)
#
# #visualize 10 random images from the loaded dataset
# # plt.figure(figsize=(12,12))
# # for i in range(10):
# #     idx = random.randint(0, len(train_dataset_untransformed))
# #     image, class_name, class_index = train_dataset_untransformed[idx]
# #     ax=plt.subplot(2,5,i+1) # create an axis
# #     ax.title.set_text(class_name + '-' + str(class_index)) # create a name of the axis based on the img name
# #     plt.imshow(image) # show the img
# # plt.show()
#
# if torch.cuda.is_available():
#     print(
#         "yes"
#     )
# else:
#     print(
#         "no"
#     )
#
#
#
print("--------------------------")



#The following is an attempt at a modified attribute loader from a tutorial
class AttributesDataset():
    def __init__(self, annotation_path):
        letter_labels = []
        # gender_labels = []
        # article_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                letter_labels.append(row['class_name'])
                # gender_labels.append(row['gender'])
                # article_labels.append(row['articleType'])

        self.letter_labels = np.unique(letter_labels)
        # self.gender_labels = np.unique(gender_labels)
        # self.article_labels = np.unique(article_labels)

        self.num_letters = len(self.letter_labels)
        # self.num_genders = len(self.gender_labels)
        # self.num_articles = len(self.article_labels)

        self.letter_id_to_name = dict(zip(range(len(self.letter_labels)), self.letter_labels))
        self.letter_name_to_id = dict(zip(self.letter_labels, range(len(self.letter_labels))))

        # self.gender_id_to_name = dict(zip(range(len(self.gender_labels)), self.gender_labels))
        # self.gender_name_to_id = dict(zip(self.gender_labels, range(len(self.gender_labels))))
        #
        # self.article_id_to_name = dict(zip(range(len(self.article_labels)), self.article_labels))
        # self.article_name_to_id = dict(zip(self.article_labels, range(len(self.article_labels))))
print("This was the attribute dataset.")
print("******************************")
#The above dataloader didn't work as expected. Here is one modified from a tutorial:
class ASLDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.letter_labels = [] #color
        # self.gender_labels = []
        # self.article_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['file_path']) #'image_path'
                self.letter_labels.append(self.attr.letter_name_to_id[row['class_name']]) #color  'baseColour'
                # self.gender_labels.append(self.attr.gender_name_to_id[row['gender']])
                # self.article_labels.append(self.attr.article_name_to_id[row['articleType']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'letter_labels': self.letter_labels[idx] #color,
                # 'gender_labels': self.gender_labels[idx],
                # 'article_labels': self.article_labels[idx]
            }
        }
        return dict_data

print("This was the ASL dataset.")
print("--------------------------")