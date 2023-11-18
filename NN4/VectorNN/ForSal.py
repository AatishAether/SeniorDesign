import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.onnx

batch_size = 1
num_workers = 6
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(threshold=torch.inf)

class ASLDataset(torch.utils.data.Dataset): # inheriting from Dataset class
    def __init__(self, csv_file, root_dir="", transform=transforms.ToTensor()):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df)

    def __getitem__(self, idx):
        dataPoints = []
        # I use range(5) here assuming that the data is in the format of: data1 data2 data3 data4 data5 label_IDX label1
        for i in range(5):
            dataPoints.append(eval(self.annotation_df.iloc[idx, i]))
        dataPoints = torch.tensor(dataPoints, device='cpu')

        # I use iloc[idx, 5] here assuming that the data is in the format of: data1 data2 data3 data4 data5 label_IDX label1 (label_IDX is at index 5)
        label = int(self.annotation_df.iloc[idx, 5])
        return dataPoints, label


train_dataset = ASLDataset('./ASL_Alph_Train.csv') #, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_dataset = ASLDataset('./ASL_Alph_Test.csv')  # val.csv
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# The label_IDX is a numerical value because it will correspond to an output here.
classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z') #

# Depending on how many classes you use, set this accordingly
num_classes = 28

class VSNetwork(nn.Module):
    def __init__(self):
        super(VSNetwork, self).__init__()

        # in_channels is set to 5 because I assume that the data is from 5 flex sensors stored in five different columns of the csv
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=12, kernel_size=3)
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
model = VSNetwork()

loss_fn = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Function to save the model
def saveModel():
    path = "./ASLTesterVecNN.pth"
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
            # print(type(labels))
            print(float(labels))
            # labels = Variable(labels.to(device))
            points = points.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using points from the training set
            outputs = model(points)
            # compute the loss based on model output and real labels
            print(type(labels))
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

    points = points.to(device)
    labels = labels.to(device)
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

if __name__ == "__main__":
    # Let's build our model
    train(2)
    print('Finished Training')
    model = VSNetwork()

    model.cuda()
    path = "ASLTesterVecNN.pth"
    model.load_state_dict(torch.load(path))

    print(testAccuracy())
    # Test with batch of points
    testBatch()
