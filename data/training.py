from getdata import Get_data
import torch
import torch.nn as nn
from torchsummary import summary


''''Build your own CNN model'''
class Net(nn.Module):
    def __init__(self,num_classes=8):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.conv3= nn.Conv2d(64,128,3)
        self.pool = nn.MaxPool2d(2,2)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,8)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x= self.pool(x)
        x = self.relu(self.conv2(x))
        x= self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.global_pool(x)
        x = x.view(x.size(0),-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Set parameters
batch_size=64
num_classes=8
size=108
datapath=r"C:/data"
epochs=100
lr=0.005


# Get data
get= Get_data(batch_size=batch_size,size=size,datapath=datapath)
trainload,valload= get.get_fish()

#Initialize model and optimizer and loss function
model=Net(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# take a look of your model
summary(model,(3,size,size))


# Training and validation
for epoch in range(epochs):
    # Training process
    model.train()
    print(f'\nEpoch: {epoch} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for i,(x,y) in enumerate(trainload):

        out = model(x)
        loss = criterion(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (out.argmax(dim=1) == y.argmax(dim=1)).sum().item() / x.size(0)
        
        if i%10==0:
            print(f'Training: , Step: {i}, Loss: {loss.item() : 5f}, Accuracy: {accuracy: 5f}')


    # Validation process
    model.eval()
    accuracies=0
    lossess=0
    with torch.no_grad():
        for i,(x,y) in enumerate(valload):

            out = model(x)
            loss = criterion(out,y)

            accuracy = (out.argmax(dim=1) == y.argmax(dim=1)).sum().item() / x.size(0)
            accuracies+=accuracy
            lossess+=loss.item()

        print(f'\nValidating: , Average Accuracy: {accuracies/len(valload): 5f}, Average Loss: {lossess/len(valload): 5f}')
