import torch
import torch.nn as nn
import torch.nn.functional as F

'''
class DQN(nn.Module): # 原DQN

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(512, action_dim)
        self.__device = device

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = F.relu(self.__fc1(x.view(x.size(0), -1)))
        return self.__fc2(x)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
'''

class DQN(nn.Module): # Dueling_DQN

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False) # 84->20
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False) # 20->9
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False) # 9->7
        self.__fc1 = nn.Linear(64*7*7, 512)
        self.__fc2 = nn.Linear(64*7*7, 512)
        self.__fc3 = nn.Linear(512, 1)
        self.__fc4 = nn.Linear(512, action_dim)
        self.__device = device
        self.__action_dim = action_dim

    def forward(self, x):
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        x = x.view(x.size(0), -1)
        x1 = self.__fc3(F.relu(self.__fc1(x))).expand(x.size(0), self.__action_dim) # 状态价值函数
        x2 = self.__fc4(F.relu(self.__fc2(x))) # 行为优势函数
        return x1+x2-x2.mean(1).unsqueeze(1).expand(x.size(0), self.__action_dim)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
