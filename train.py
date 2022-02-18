### Model Training ###

from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset import DatasetAudio
import pandas as pd
from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm
from time import sleep


# Load data df
df_positives = pd.read_csv('/content/drive/MyDrive/BackApp/backapp_positive_audios_with_path.csv')
df_negatives = pd.read_csv('/content/drive/MyDrive/BackApp/backapp_negative_audios_with_path.csv')
# df_negatives = df_negatives.sample(n=1000)

# Create 3 different datasets from original df
audio_dataset_1 = DatasetAudio(df_positives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')
audio_dataset_2 = DatasetAudio(df_positives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')
audio_dataset_3 = DatasetAudio(df_positives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')
audio_dataset_4 = DatasetAudio(df_positives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')
audio_dataset_5 = DatasetAudio(df_positives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')
audio_dataset_6 = DatasetAudio(df_positives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')
audio_dataset_negative = DatasetAudio(df_negatives, '/content/drive/MyDrive/BackApp/data/backapp_full_audios/')

audio_dataset = ConcatDataset([audio_dataset_1, audio_dataset_2, audio_dataset_3, audio_dataset_4, audio_dataset_5, audio_dataset_6, audio_dataset_negative])

# Define the number of samples for training and validation - 90% and 10% respectively
train_size = int(0.9 * len(audio_dataset))
val_size = len(audio_dataset) - train_size

# Randomly split the dataset into training and validation sets
train_dataset, val_dataset = random_split(audio_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class DistressModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=(2, 2), padding=(2, 2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(10304, 50)
        self.fc2 = nn.Linear(50, 2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv4(x)), 2))
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)
    
model = DistressModel().to(device)


# cost function used to determine best parameters
cost = torch.nn.CrossEntropyLoss()

# used to create optimal parameters
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the training function

batch_size = 64

def train(dataloader, model, epochs):
    model.train()
    size = len(dataloader.dataset)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(dataloader)),
                                                    epochs=epochs,
                                                    anneal_strategy='linear')

    
    for epoch in range(epochs):
      with tqdm(dataloader, unit="batch") as tepoch:
        for batch, (X, Y) in enumerate(tepoch):

          tepoch.set_description(f"Epoch {epoch}")
          X, Y = X.to(device), Y.to(device)
          X_m, X_s = X.mean(), X.std()
          X = (X - X_m) / X_s
          optimizer.zero_grad()
          pred = model(X)
          # Y = Y.unsqueeze(1)
          # Y = Y.float()
          loss = cost(pred, Y)

          correct = (pred.argmax(1)==Y).type(torch.float).sum().item()
          accuracy = correct / batch_size

          loss.backward()
          # print(model.conv1.weight.grad) 
          # print(model.conv2.bias.grad)
          optimizer.step()
          scheduler.step()

          tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
          sleep(0.1)

          # if batch % 10 == 0:
          #     loss, current = loss.item(), batch * len(X)
          #     print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    total_pred = 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            X_m, X_s = X.mean(), X.std()
            X = (X - X_m) / X_s
            pred = model(X)

            correct += (pred.argmax(1)==Y).type(torch.float).sum().item()
            total_pred += pred.shape[0]

    acc = correct / total_pred

    print(f'Accuracy: {acc:.2f}, Total items: {total_pred}')

            
epochs = 70

train(train_loader, model, epochs)
test(val_loader, model)

torch.save(model.state_dict(), "best_model.pt")

model.eval()
test_loss, correct = 0, 0

with torch.no_grad():
    for batch, (X, Y) in enumerate(val_loader):
        X, Y = X.to(device), Y.to(device)
        X_m, X_s = X.mean(), X.std()
        X = (X - X_m) / X_s
        pred = model(X)
        print("Predicted:")
        print(f"{pred.argmax(1)}")
        print("Actual:")
        print(f"{Y}")
        break


# # ------------------------------------------------------- #
# # Model
# class DistressClassifier(nn.Module):
#     # ----------------------------
#     # Build the model architecture
#     # ----------------------------
#     def __init__(self):
#         super().__init__()
#         conv_layers = []

#         # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
#         self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
#         self.relu1 = nn.ReLU()
#         self.bn1 = nn.BatchNorm2d(8)
#         # init.kaiming_normal_(self.conv1.weight, a=0.1)
#         self.conv1.bias.data.zero_()
#         conv_layers += [self.conv1, self.relu1, self.bn1]

#         # Second Convolution Block
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu2 = nn.ReLU()
#         self.bn2 = nn.BatchNorm2d(16)
#         # init.kaiming_normal_(self.conv2.weight, a=0.1)
#         self.conv2.bias.data.zero_()
#         conv_layers += [self.conv2, self.relu2, self.bn2]

#         # Second Convolution Block
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu3 = nn.ReLU()
#         self.bn3 = nn.BatchNorm2d(32)
#         # init.kaiming_normal_(self.conv3.weight, a=0.1)
#         self.conv3.bias.data.zero_()
#         conv_layers += [self.conv3, self.relu3, self.bn3]

#         # Second Convolution Block
#         self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         self.relu4 = nn.ReLU()
#         self.bn4 = nn.BatchNorm2d(64)
#         # init.kaiming_normal_(self.conv4.weight, a=0.1)
#         self.conv4.bias.data.zero_()
#         conv_layers += [self.conv4, self.relu4, self.bn4]

#         # Linear Classifier
#         self.ap = nn.AdaptiveAvgPool2d(output_size=1)
#         self.lin = nn.Linear(in_features=64, out_features=1)

#         # Wrap the Convolutional Blocks
#         self.conv = nn.Sequential(*conv_layers)
 
#     # ----------------------------
#     # Forward pass computations
#     # ----------------------------
#     def forward(self, x):
#         # Run the convolutional blocks
#         x = self.conv(x)

#         # Adaptive pool and flatten for input to linear layer
#         x = self.ap(x)
#         x = x.view(x.shape[0], -1)

#         # Linear layer
#         x = self.lin(x)

#         # Final output
#         return x

# # ----------------------------------------------------------- #    


# # Create the model and put it on the GPU if available
# myModel = DistressClassifier()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# myModel = myModel.to(device)
# # Check that it is on Cuda
# next(myModel.parameters()).device

# # ----------------------------
# # Training Loop
# # ----------------------------
# def training(model, train_dl, num_epochs):
#   # Loss Function, Optimizer and Scheduler
#   criterion = nn.BCEWithLogitsLoss()
#   optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
#   scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
#                                                 steps_per_epoch=int(len(train_dl)),
#                                                 epochs=num_epochs,
#                                                 anneal_strategy='linear')

#   # Repeat for each epoch
#   for epoch in range(num_epochs):
#     running_loss = 0.0
#     correct_prediction = 0
#     total_prediction = 0

#     # Repeat for each batch in the training set
#     for i, data in enumerate(train_dl):
#         # Get the input features and target labels, and put them on the GPU
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # Normalize the inputs
#         # inputs_m, inputs_s = inputs.mean(), inputs.std()
#         # inputs = (inputs - inputs_m) / inputs_s
        
#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(inputs)
#         print(outputs)
#         labels = labels.unsqueeze(1)
#         labels = labels.float()
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         scheduler.step()

#         # Keep stats for Loss and Accuracy
#         running_loss += loss.item()

#         # Get the predicted class with the highest score
#         _, prediction = torch.max(outputs,1)
#         print(prediction)
#         # Count of predictions that matched the target label
#         correct_prediction += (prediction == labels).sum().item()
#         total_prediction += prediction.shape[0]

#         if i % 10 == 0:    # print every 10 mini-batches
#            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
#     # Print stats at the end of the epoch
#     num_batches = len(train_dl)
#     avg_loss = running_loss / num_batches
#     acc = correct_prediction/total_prediction
#     print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

#   print('Finished Training')
  
# num_epochs=2   # Just for demo, adjust this higher.
# training(myModel, train_loader, num_epochs)