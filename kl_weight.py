import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchbnn as bnn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(args):
    # Create directories if they don't exist
    os.makedirs(args.ck_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)

    # Function to get filtered MNIST datasets
    def getSets(filteredClass=None, removeFiltered=True):
        train = torchvision.datasets.MNIST('./data/', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
        test = torchvision.datasets.MNIST('./data/', train=False, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                           ]))

        if filteredClass is not None:
            train_loader = torch.utils.data.DataLoader(train, batch_size=len(train))
            train_labels = next(iter(train_loader))[1].squeeze()
            test_loader = torch.utils.data.DataLoader(test, batch_size=len(test))
            test_labels = next(iter(test_loader))[1].squeeze()

            if removeFiltered:
                trainIndices = torch.nonzero(train_labels != filteredClass).squeeze()
                testIndices = torch.nonzero(test_labels != filteredClass).squeeze()
            else:
                trainIndices = torch.nonzero(train_labels == filteredClass).squeeze()
                testIndices = torch.nonzero(test_labels == filteredClass).squeeze()

            train = torch.utils.data.Subset(train, trainIndices)
            test = torch.utils.data.Subset(test, testIndices)

        return train, test

    # Define Bayesian Neural Network model
    class BayesianMnistNet(nn.Module):
        def __init__(self):
            super(BayesianMnistNet, self).__init__()
            self.conv1 = bnn.BayesConv2d(prior_mu=0., prior_sigma=.1, in_channels=1, out_channels=32, kernel_size=3, padding=1)
            self.conv2 = bnn.BayesConv2d(prior_mu=0., prior_sigma=.1, in_channels=32, out_channels=64, kernel_size=3, padding=1)
            self.fc1 = bnn.BayesLinear(prior_mu=0., prior_sigma=.1, in_features=64 * 7 * 7, out_features=128)
            self.fc2 = bnn.BayesLinear(prior_mu=0., prior_sigma=.1, in_features=128, out_features=10)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Evaluation function
    def evaluate(model, test_loader_seen, test_loader_unseen, device, result_dir, kl_weight):
        nruntest = 32  # Number of test runs

        model.eval()
        with torch.no_grad():
            # Evaluate seen class
            imgs, labels = next(iter(test_loader_seen))
            imgs, labels = imgs.to(device), labels.to(device)

            test_result_samples_seen = torch.zeros(nruntest, len(imgs), 10)
            for i in range(nruntest):
                test_result_samples_seen[i] = F.softmax(model(imgs), dim=1)

            sampleMean_seen = torch.mean(test_result_samples_seen, dim=(0, 1))
            sampleStd_seen = torch.std(test_result_samples_seen, dim=(0, 1))

            # Save a sample test image for qualitative analysis
            test_image_path_seen = os.path.join(result_dir, f'sample_test_image_seen.png')
            torchvision.utils.save_image(imgs.cpu(), test_image_path_seen)

            # Evaluate unseen class
            imgs, labels = next(iter(test_loader_unseen))
            imgs, labels = imgs.to(device), labels.to(device)

            test_result_samples_unseen = torch.zeros(nruntest, len(imgs), 10)
            for i in range(nruntest):
                test_result_samples_unseen[i] = F.softmax(model(imgs), dim=1)

            sampleMean_unseen = torch.mean(test_result_samples_unseen, dim=(0, 1))
            sampleStd_unseen = torch.std(test_result_samples_unseen, dim=(0, 1))

            # Save a sample test image(unseen) for qualitative analysis
            test_image_path_unseen = os.path.join(result_dir, f'sample_test_image_unseen.png')
            torchvision.utils.save_image(imgs.cpu(), test_image_path_unseen)

        # Plot and save results
        plt.style.use('ggplot')
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.title(f'Mean (kl_weight={kl_weight})')
        bar_width = 0.4
        bar_positions = np.arange(10)
        plt.bar(bar_positions - bar_width / 2, sampleMean_seen.numpy(), width=bar_width, label='seen class(5)')
        plt.bar(bar_positions + bar_width / 2, sampleMean_unseen.numpy(), width=bar_width, label='unseen class(4)')
        plt.xlabel('digits')
        plt.ylabel('digit prob')
        plt.ylim([0, 1])
        plt.xticks(np.arange(10))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(f'Std (kl_weight={kl_weight})')
        plt.bar(bar_positions - bar_width / 2, sampleStd_seen.numpy(), width=bar_width, label='seen class(5)')
        plt.bar(bar_positions + bar_width / 2, sampleStd_unseen.numpy(), width=bar_width, label='unseen class(4)')
        plt.xlabel('digits')
        plt.ylabel('std')
        plt.xticks(np.arange(10))
        plt.legend()

        result_image_path = os.path.join(result_dir, f'OOD_kl_weight_{kl_weight}.png')
        plt.savefig(result_image_path)
        plt.close()


    # Get filtered MNIST dataset for seen and unseen classes
    train_filtered_seen, test_filtered_seen = getSets(filteredClass=5, removeFiltered=False)
    train_filtered_unseen, test_filtered_unseen = getSets(filteredClass=4, removeFiltered=False)

    # DataLoaders for evaluation
    test_loader_seen = DataLoader(test_filtered_seen, batch_size=1, shuffle=True)
    test_loader_unseen = DataLoader(test_filtered_unseen, batch_size=1, shuffle=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train:
        # Get filtered MNIST dataset for training, excluding class '4'
        train, _ = getSets(filteredClass=4)
        train_loader = DataLoader(train, batch_size=64, shuffle=True)

        # Training loop
        for kl_weight in [1, 0.1, 0.01, 0.001]:
            model = BayesianMnistNet().to(device)
            ce = nn.CrossEntropyLoss()
            kl = bnn.BKLLoss(reduction='sum', last_layer_only=args.last_layer_only)
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(args.epoch):
                model.train()
                epoch_loss = 0.0

                with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epoch}', leave=True) as pbar:
                    for batch_id, (imgs, labels) in enumerate(train_loader):
                        imgs, labels = imgs.to(device), labels.to(device)
                        optimizer.zero_grad()

                        pred = model(imgs)
                        ce_loss = ce(pred, labels)
                        kl_loss = kl(model)
                        total_loss = ce_loss + kl_weight * kl_loss

                        total_loss.backward()
                        optimizer.step()

                        epoch_loss += total_loss.item()
                        pbar.set_postfix({
                            'KL Loss': f'{kl_loss.item():.4f}',
                            'Total Loss': f'{total_loss.item() / len(imgs):.4f}'
                        })
                        pbar.update(1)

                print(f'Epoch {epoch+1}/{args.epoch} - Epoch Loss: {epoch_loss / len(train_loader):.4f}')

            # Save model checkpoint
            ck_path = os.path.join(args.ck_dir, f'bnn_v1_kl_weight_{kl_weight}.pt')
            torch.save(model.state_dict(), ck_path)

        print('Training finished.')

    else:
        # Evaluate the model for each kl_weight value
        for kl_weight in [1, 0.1, 0.01, 0.001]:
            # Load pre-trained model for evaluation
            model = BayesianMnistNet().to(device)
            model_path = os.path.join(args.ck_dir, f'bnn_v1_kl_weight_{kl_weight}.pt')
            model.load_state_dict(torch.load(model_path))

            # Evaluate seen and unseen classes
            evaluate(model, test_loader_seen, test_loader_unseen, device, args.result_dir, kl_weight)

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Bayesian Deep Learning with Hyper-parameter Testing')
    parser.add_argument('--ck_dir', type=str, default='./checkpoint/OOD/kl_weight', help='Directory to save model checkpoints')
    parser.add_argument('--result_dir', type=str, default='./result/OOD/kl_weight', help='Directory to save result files')
    parser.add_argument('--train', action='store_true', help='Whether to train the model')
    parser.add_argument('--epoch', type=int, default=3, help='Number of epochs to train')
    parser.add_argument('--last_layer_only', action='store_true', help='Whether to apply KL divergence only to the last layer')
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
