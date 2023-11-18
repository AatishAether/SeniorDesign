import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.onnx

batch_size = 1
num_workers = 6
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)
class ASLDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    def __init__(self, csv_file, root_dir="", transform=transforms.ToTensor()):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir # root directory of points, leave "" if using the image path column in the __getitem__ method
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        dataPoints = []
        # print(self.annotation_df.iloc[idx, :20])
        for i in range(20):
            dataPoints.append(torch.tensor(eval(self.annotation_df.iloc[idx, i]), dtype=torch.float64))
        # print(dataPoints)
        # image_path = os.path.join(self.root_dir, self.annotation_df.iloc[idx, 22]) #use image path column (index = 1) in csv file
        # image = read_image(image_path)
        print(dataPoints)
        # print("Got item")

        label = self.annotation_df.iloc[idx, 21]
        return dataPoints, label #class_index , class_name

train_dataset = ASLDataset('./ASL_Alph_Train.csv') #, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = ASLDataset('./ASL_Alph_Test.csv')  # val.csv
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z')

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        att = torch.sigmoid(self.conv1(x))
        return x * att

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # Layer 1: Convolutional layer
        self.conv1 = nn.Conv1d(in_channels=20, out_channels=60, kernel_size=3)
        self.attention1 = SpatialAttention(in_channels=60)
        self.bn1 = nn.BatchNorm2d(60)
        self.relu1 = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Layer 2: Convolutional layer
        self.conv2 = nn.Conv3d(in_channels=60, out_channels=120, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(120)
        self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Layer 3: Fully connected layer
        self.fc1 = nn.Linear(360000, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # Layer 4: Output layer
        self.fc2 = nn.Linear(512, 28)

    def forward(self, x):
        # x = torch.tensor(x, dtype=torch.float32)
        # x = x.unsqueeze(1)
        # print(x)
        # new_x = []
        # for i in x:
        #     print(i)
        #     new_x.append(torch.tensor([tensor for tensor in i], dtype=torch.double))
        #
        #
        #
        # print(new_x)
        #
        # print(torch.stack(new_x))
        # x = torch.stack(new_x)
        x = torch.stack(x)
        # x = x.unsqueeze(3)
        print(x)

        # Layer 1
        x = self.conv1(x)
        x = self.attention1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = self.maxpool1(x)

        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.maxpool2(x)

        # Flatten tensor
        x = x.view(x.size(0), -1)

        # Layer 3
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)

        # Layer 4
        x = self.fc2(x)

        return x


# Instantiate a neural network model
model = Network()

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# Function to save the model
def saveModel():
    path = "./ASLVecNN.pth"
    torch.save(model.state_dict(), path)


# Function to test the model with the test dataset and print the accuracy for the test points
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in val_dataloader:
            points, labels = data

            #Define the device which will be used for processing
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            #Modify both the points and the labels so that they are stored as tensors
            points = points.to(device)
            labels = labels.to(device)

            # run the model on the test set to predict labels
            outputs = model(points)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test points
    accuracy = (100 * accuracy / total)
    return (accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (points, labels) in enumerate(train_dataloader, 0):

            # get the inputs
            # points = Variable(points.to(device))
            print(type(labels))
            # labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using points from the training set
            outputs = model(points)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 points
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test points
        accuracy = testAccuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Function to show the points
def pointshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of points and show the labels predictions
def testBatch():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get batch of points from the test DataLoader
    points, labels = next(iter(val_dataloader))

    print(points)
    # show all points as one image grid
    # pointshow(torchvision.utils.make_grid(points))
    # points = points.to(device)
    # labels = labels.to(device)
    # Show the real labels on the screen
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
                                    for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(points)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs.data, 1)

    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))

#Function to Convert to ONNX
def Convert_ONNX():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 300, 300)
    dummy_input = dummy_input.to(device)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ASLVecNN_O1.onnx")       # where to save the model
    print(" ")
    print('Model has been converted to ONNX')

def Convert_ONNX2():
    device = torch.device("cpu")
    model.to(device)
    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 3, 300, 300)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ASLVecNNSpat.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


if __name__ == "__main__":
    # Let's build our model
    train(2)
    print('Finished Training')
    model = Network()
    # Test which classes performed well
    # print(testAccuracy())

    # Let's load the model we just created and test the accuracy per label

    model.cuda()
    path = "ASLVecNN.pth"
    model.load_state_dict(torch.load(path))

    print(testAccuracy())
    # Test with batch of points
    testBatch()


    # Convert_ONNX()
    Convert_ONNX2()
