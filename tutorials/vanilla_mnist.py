import time
import sys
import torch
import torchvision
import torch.nn as nn

# Works extremely fast on gpu (first epoch already at 95%)

BATCH_SIZE = 128 
EPOCHS = 1

train_dataset = torchvision.datasets.MNIST(root='~',train = True,transform=torchvision.transforms.ToTensor())
train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)

test_dataset = torchvision.datasets.MNIST(root='~',train=True,transform = torchvision.transforms.ToTensor())
test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=1)

print('Train Dataset Contains {} Examples'.format(len(train_dataset)))
print('Test Dataset Contains {} Examples'.format(len(test_dataset)))

device = torch.device("cpu")

class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet,self).__init__()
        self.conv1 = nn.Conv2d(1,256,stride=2,kernel_size=4)
        self.conv2 = nn.Conv2d(256,128,stride=2,kernel_size=4)

        self.linear = nn.Linear(128*5*5,100)
        self.linear2 = nn.Linear(100,10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(-1,128*5*5)
        x = self.linear(x)
        x = self.linear2(x)

        return x




model = simpleNet().to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

for epoch in range(EPOCHS):
    
    for img, label in train_data_loader:
        
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        
        loss = loss_fn(out,label)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _,idx = out.max(dim=-1)
        num_correct = 0
        for i in range(idx.shape[0]):
            if idx[i] == label[i]:
                num_correct += 1
        print(str(num_correct) + ' / ' + str(idx.shape[0]))
        print(num_correct / idx.shape[0])
