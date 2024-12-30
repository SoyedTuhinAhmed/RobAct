import torch
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

class gRReLU(nn.Module):
    def __init__(self, mean_scale=0.9, std_scale=0.1, mean_shift=0.1, 
                 std_shift=0.1,  inplace=False):
        super(gRReLU, self).__init__()
        self.mu_scale, self.mu_shift = nn.Parameter(torch.tensor(mean_scale)), nn.Parameter(torch.tensor(mean_shift))
        self.std_scale, self.std_shift = nn.Parameter(torch.tensor(std_scale)), nn.Parameter(torch.tensor(std_shift))
        self.inplace = inplace

    def forward(self, x):
        if self.training:
            # Sample negative slope 'a' from a Gaussian distribution N(mean, std)
            eps_scale = torch.randn(x.shape, device=x.device)
            eps_shift = torch.randn(x.shape, device=x.device)
            return torch.where(x >= 0, x, (x*eps_scale*self.std_scale.to(x.device) + self.mu_scale.to(x.device))+ self.std_shift.to(x.device)*eps_shift + self.mu_shift.to(x.device))
        else:
            # During evaluation, use the mean value of the Gaussian distribution
            return F.leaky_relu(x, negative_slope=((self.std_scale + self.mu_scale)+ self.std_shift + self.mu_shift), inplace=self.inplace)
        
class gRReLU_pos(nn.Module):
    def __init__(self, mean=1.0, std=0.1, inplace=False):
        super(gRReLU_pos, self).__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, x):
        if self.training:
            # Sample positive multiplier 'a' from a Gaussian distribution N(mean, std)
            a = torch.empty_like(x).normal_(self.mean, self.std)
            return torch.where(x >= 0, a * x, torch.zeros_like(x))
        else:
            # During evaluation, use standard ReLU with mean multiplier
            return torch.where(x >= 0, x * self.mean, torch.zeros_like(x))

class Model:
    class Net(nn.Module):
        def __init__(self, activation_type='relu'):
            super(Model.Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.bn1 = nn.BatchNorm2d(6)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)
            # Adjusted for MNIST input size
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
            self.bn3 = nn.BatchNorm1d(120)
            self.fc2 = nn.Linear(120, 84)
            self.bn4 = nn.BatchNorm1d(84)
            self.fc3 = nn.Linear(84, 10)

            if activation_type == 'relu':
                self.activation = nn.ReLU()
            elif activation_type == 'grrelu':
                self.activation = gRReLU()
            elif activation_type == 'grrelu_pos':
                self.activation = gRReLU_pos()

        def forward(self, x):
            x = self.bn1(self.conv1(x))
            x = self.activation(x)
            x = F.max_pool2d(x, (2, 2))

            x = self.bn2(self.conv2(x))
            x = self.activation(x)
            x = F.max_pool2d(x, 2)

            x = x.view(-1, self.num_flat_features(x))

            x = self.bn3(self.fc1(x))
            x = self.activation(x)

            x = self.bn4(self.fc2(x))
            x = self.activation(x)

            x = self.fc3(x)
            return x

        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features

    def __init__(self, train_loader, test_loader, criterion, optimizer, num_epochs=5, activation_type='relu'):
        self.net = self.Net(activation_type)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.best_accuracy = 0.0
        self.activation_type = activation_type

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
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
                        f'[{self.activation_type}] Epoch [{epoch + 1}/{self.num_epochs}], Step [{i + 1}/{len(self.train_loader)}], Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0
            # Step the scheduler
            scheduler.step()
            # Evaluate and print accuracy after each epoch
            accuracy = self.eval(self.test_loader)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                torch.save(self.net.state_dict(), f'./best_model_{self.activation_type}.pth')
                print(
                    f'[{self.activation_type}] Saved best model with accuracy: {self.best_accuracy:.2f}%')
        print(f'[{self.activation_type}] Finished Training')
        print(f'[{self.activation_type}] Best accuracy during training: {self.best_accuracy:.2f}%')

    def eval(self, test_loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
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
        print(f'[{self.activation_type}] Accuracy of the network on the test images: {accuracy:.2f}%')
        return accuracy

def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
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
            if isinstance(module, (nn.Conv2d, nn.Linear)):
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

    # First train with ReLU
    net_relu = Model(train_loader, test_loader, criterion, optim.AdamW, num_epochs=num_epochs, activation_type='relu')
    optimizer_relu = optim.AdamW(net_relu.net.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999))
    net_relu.optimizer = optimizer_relu
    
    # Save initial state
    initial_state = copy.deepcopy(net_relu.net.state_dict())
    
    # # Train ReLU model
    # net_relu.train()
    
    # Then train with gRReLU using same initialization
    net_grrelu = Model(train_loader, test_loader, criterion, optim.AdamW, num_epochs=num_epochs, activation_type='grrelu')
    net_grrelu.net.load_state_dict(initial_state)
    optimizer_grrelu = optim.AdamW(net_grrelu.net.parameters(), lr=0.001, weight_decay=1e-5, betas=(0.9, 0.999))
    net_grrelu.optimizer = optimizer_grrelu
    
    # # Train gRReLU model
    net_grrelu.train()
    
    print("\nFinal Results:")
    print(f"ReLU Best Accuracy: {net_relu.best_accuracy:.2f}%")
    print(f"gRReLU Best Accuracy: {net_grrelu.best_accuracy:.2f}%")
    
    # Load best models and evaluate with noise
    print("\nEvaluating models with noise (alpha=0.1):")
    
    # Set seed for reproducible noise injection
    set_seed(42)
    
    # Lists to store results
    alphas = np.arange(0, 1.6, 0.2)
    K = 10  # Number of evaluations per alpha
    relu_accuracies_all = []
    grrelu_accuracies_all = []
    
    print("\nEvaluating models with different noise levels:")
    for alpha in alphas:
        relu_accuracies_alpha = []
        grrelu_accuracies_alpha = []
        
        print(f"Alpha={alpha:.1f}:")
        for k in range(K):
            # Evaluate ReLU model

            # Load models
            net_relu.net.load_state_dict(torch.load(f'./best_model_relu.pth', weights_only=True))
            net_grrelu.net.load_state_dict(torch.load(f'./best_model_grrelu.pth', weights_only=True))
            # net_grrelu.net.load_state_dict(torch.load(f'./best_model_grrelu_pos.pth', weights_only=True))

            net_relu_copy = copy.deepcopy(net_relu.net)
            inject_noise(net_relu_copy, alpha=alpha)
            net_relu.net = net_relu_copy
            relu_acc = net_relu.eval(test_loader)
            relu_accuracies_alpha.append(relu_acc)
            
            # Evaluate gRReLU model
            net_grrelu_copy = copy.deepcopy(net_grrelu.net)
            inject_noise(net_grrelu_copy, alpha=alpha)
            net_grrelu.net = net_grrelu_copy
            grrelu_acc = net_grrelu.eval(test_loader)
            grrelu_accuracies_alpha.append(grrelu_acc)
            
            if (k + 1) % 10 == 0:
                print(f"  Completed {k + 1}/{K} evaluations of alpha={alpha:.1f}")
        
        relu_accuracies_all.append(relu_accuracies_alpha)
        grrelu_accuracies_all.append(grrelu_accuracies_alpha)
        
        print(f"  ReLU Mean Accuracy: {np.mean(relu_accuracies_alpha):.2f}% ± {np.std(relu_accuracies_alpha):.2f}%")
        print(f"  gRReLU Mean Accuracy: {np.mean(grrelu_accuracies_alpha):.2f}% ± {np.std(grrelu_accuracies_alpha):.2f}%")

    # Calculate means and standard deviations
    relu_means = np.array([np.mean(accs) for accs in relu_accuracies_all])
    relu_stds = np.array([np.std(accs) for accs in relu_accuracies_all])
    grrelu_means = np.array([np.mean(accs) for accs in grrelu_accuracies_all])
    grrelu_stds = np.array([np.std(accs) for accs in grrelu_accuracies_all])

    torch.save(relu_accuracies_all, './relu_accuracies_all.pth')
    torch.save(grrelu_accuracies_all, './grrelu_accuracies_all.pth')

    plt.figure(figsize=(10, 6))
    # Plot ReLU with error band
    plt.plot(alphas, relu_means, 'b-', label='ReLU')
    plt.fill_between(alphas, relu_means - relu_stds, relu_means + relu_stds, color='b', alpha=0.2)
    # Plot gRReLU with error band
    plt.plot(alphas, grrelu_means, 'r-', label='gRReLU')
    plt.fill_between(alphas, grrelu_means - grrelu_stds, grrelu_means + grrelu_stds, color='r', alpha=0.2)
    
    plt.xlabel('Noise Level (α)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy vs Noise Level (with ±1 std)')
    plt.legend()
    plt.grid(True)
    plt.savefig('noise_robustness.png')
    plt.close()

if __name__ == "__main__":
    main()
