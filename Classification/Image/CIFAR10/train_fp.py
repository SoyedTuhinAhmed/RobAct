import torch, noise_model
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random, copy, numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can choose any seed value you prefer

# class gRReLU(nn.Module):
#     def __init__(self, mean=0.2, std=0.03, inplace=False):
#         super(gRReLU, self).__init__()
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace

#     def forward(self, x):
#         if self.training:
#             # Sample negative slope 'a' from a Gaussian distribution N(mean, std)
#             a = torch.empty_like(x).normal_(self.mean, self.std)
#             return torch.where(x >= 0, x, a * x)
#         else:
#             # During evaluation, use the mean value of the Gaussian distribution
#             return F.leaky_relu(x, negative_slope=self.mean, inplace=self.inplace)
#             # return F.relu(x, inplace=self.inplace)

# class gRReLU_pos(nn.Module):
#     def __init__(self, mean=1.0, std=0.1, inplace=False):
#         super(gRReLU_pos, self).__init__()
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace

#     def forward(self, x):
#         if self.training:
#             # Sample positive multiplier 'a' from a Gaussian distribution N(mean, std)
#             a = torch.empty_like(x).normal_(self.mean, self.std)
#             return torch.where(x >= 0, a * x, torch.zeros_like(x))
#         else:
#             # During evaluation, use standard ReLU with mean multiplier
#             return torch.where(x >= 0, x * self.mean, torch.zeros_like(x))

class gRReLU(nn.Module):
    def __init__(self, mean_scale=.97, std_scale=.01, mean_shift=0.03, 
                 std_shift=0.01, inplace = True, learnable_mu=True, learnable_std=True):
        super(gRReLU, self).__init__()
        if learnable_std:
            self.raw_std_scale = nn.Parameter(torch.log(torch.tensor(std_scale)))
            self.raw_std_shift = nn.Parameter(torch.log(torch.tensor(std_shift)))
        else:
            self.raw_std_scale = torch.log(torch.tensor(std_scale))
            self.raw_std_shift = torch.log(torch.tensor(std_shift))

        if learnable_mu:
            self.mu_scale = nn.Parameter(torch.tensor(mean_scale))
            self.mu_shift = nn.Parameter(torch.tensor(mean_shift))
        else:
            self.mu_scale = torch.tensor(mean_scale)
            self.mu_shift = torch.tensor(mean_shift)
        self.inplace = inplace

    def forward(self, x):
        std_scale = torch.exp(self.raw_std_scale.to(x.device))
        std_shift = torch.exp(self.raw_std_shift.to(x.device))
        if self.training:
            # Sample negative slope 'a' from a Gaussian distribution N(mean, std)
            eps_scale = torch.randn(x.shape, device=x.device)
            eps_shift = torch.randn(x.shape, device=x.device)
            # x_noise = (x*eps_scale*self.std_scale.to(x.device) + self.mu_scale.to(x.device))+ self.std_shift.to(x.device)*eps_shift + self.mu_shift.to(x.device)
            # return F.relu(x_noise, inplace=self.inplace)
            return torch.where(x >= 0, x, (0.1*(eps_scale*std_scale + self.mu_scale.to(x.device))) + (std_shift*eps_shift + self.mu_shift.to(x.device)) )#, x*0.1)
        else:
            # During evaluation, use the mean value of the Gaussian distribution
            # return F.leaky_relu(x, negative_slope=0.1, inplace=self.inplace)
            return torch.where(x >= 0, x, (0.1*self.mu_scale.to(x.device)) + self.mu_shift.to(x.device))#, x*0.1)
        
            # eps_scale = torch.randn(x.shape, device=x.device)
            # eps_shift = torch.randn(x.shape, device=x.device)
            # return torch.where(x >= 0, (x*eps_scale*self.std_scale.to(x.device) + self.mu_scale.to(x.device))+ self.std_shift.to(x.device)*eps_shift + self.mu_shift.to(x.device), x*0.1)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, activation_type='relu'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if activation_type == 'relu':
            self.conv1_act = nn.ReLU(inplace=True)
            self.conv2_act = nn.ReLU(inplace=True)
        else:
            self.conv1_act = gRReLU()
            self.conv2_act = gRReLU()
            self.conv3_act = gRReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.activation_type = activation_type

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.activation_type == 'relu':
            out = self.bn1(out)
            out = self.conv1_act(out)
        else:
            out = self.conv1_act(out)
            out = self.bn1(out)

        out = self.conv2(out)
        if self.activation_type == 'relu':
            out = self.bn2(out)
        else:
            out = self.conv2_act(out)
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            if self.activation_type != 'relu':
                identity = self.conv3_act(identity)

        out += identity
        if self.activation_type == 'relu':
            out = self.conv2_act(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, activation_type='relu'):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation_type == 'grrelu':
            self.activation = gRReLU()
        elif activation_type == 'grrelu_pos':
            self.activation = gRReLU_pos()
        self.activation_type = activation_type
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, activation_type=activation_type)
        self.layer2 = self._make_layer(128, 2, stride=2, activation_type=activation_type)
        self.layer3 = self._make_layer(256, 2, stride=2, activation_type=activation_type)
        self.layer4 = self._make_layer(512, 2, stride=2, activation_type=activation_type)
        
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride=1, activation_type='relu'):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample, activation_type))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, activation_type=activation_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.activation_type == 'relu':
            x = self.bn1(x)
            x = self.activation(x)
        else:
            x = self.activation(x)
            x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Model:
    Net = ResNet18

    def __init__(self, train_loader, test_loader, criterion, optimizer, num_epochs=5, activation_type='relu'):
        self.net = self.Net(activation_type=activation_type)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.best_accuracy = 0.0
        self.activation_type = activation_type

    def train(self, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.net.train()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(
                        f'[{self.activation_type}] Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {running_loss / 100:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
                    running_loss = 0.0
            # Step the scheduler
            scheduler.step()
            # Evaluate and print accuracy after each epoch
            accuracy = self.eval(self.test_loader, device=device)
            self.net.train()
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save(self.net.state_dict(), f'./best_model_{self.activation_type}.pth')
                print(
                    f'[{self.activation_type}] Saved best model with accuracy: {self.best_accuracy:.2f}%')
            print(f'[{self.activation_type}] Eval Accuracy: {accuracy:.2f}%, best: {self.best_accuracy:.2f}%')
            
        print(f'[{self.activation_type}] Finished Training')
        print(f'[{self.activation_type}] Best accuracy during training: {self.best_accuracy:.2f}%')

    def eval(self, test_loader, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy

def load_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader

def inject_noise(model, alpha):
    """
    Injects Gaussian noise into the weights of convolutional and linear layers only.
    Args:
        model: The neural network model
        alpha: Float between 0 and 1 indicating the noise level (e.g., 0.25 for 25% variation)
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)): # (nn.Conv2d, nn.Linear)):
                # Only inject noise into weights, not biases
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    std = torch.abs(weight) * alpha
                    noise = torch.randn_like(weight) * std
                    module.weight.data += noise

def main():
    train_loader, test_loader = load_data()
    criterion = nn.CrossEntropyLoss()
    num_epochs = 200
    nm = noise_model.RRAMNonidealities()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # First train with ReLU
    net_relu = Model(train_loader, test_loader, criterion, optim.AdamW, num_epochs=num_epochs, activation_type='relu')
    # optimizer_relu = optim.AdamW(net_relu.net.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999))
    optimizer_relu = optim.SGD(net_relu.net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    net_relu.optimizer = optimizer_relu
    
    # Save initial state
    initial_state = copy.deepcopy(net_relu.net.state_dict())
    
    # # Train ReLU model
    # net_relu.train()
    
    # Then train with gRReLU using same initialization
    net_grrelu = Model(train_loader, test_loader, criterion, optim.AdamW, num_epochs=num_epochs, activation_type='grrelu')
    # net_grrelu.net.load_state_dict(initial_state)
    # optimizer_grrelu = optim.AdamW(net_grrelu.net.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999))
    optimizer_grrelu = optim.SGD(net_grrelu.net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    net_grrelu.optimizer = optimizer_grrelu
    
    # # Train gRReLU model
    net_grrelu.train(device)

if __name__ == "__main__":
    main()