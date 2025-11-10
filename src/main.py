import torch
from torch import nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from discriminator import Discriminator
from generator import Generator
import os
from datetime import datetime

torch.manual_seed(111)

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Create results directory
def create_results_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Data setup
def load_data(batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = torchvision.datasets.MNIST(
        root=".", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    return train_loader

# Visualization function
def plot_samples(samples, title="Samples", save_path=None):
    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(samples[i].reshape(28, 28), cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.show()
    return fig

# Training function
def train_gan(num_epochs=50, batch_size=32, lr=0.0001):
    results_dir = create_results_dir()
    train_loader = load_data(batch_size)
    
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    
    # Plot real samples
    real_samples, _ = next(iter(train_loader))
    plot_samples(real_samples, "Real MNIST Samples", f"{results_dir}/real_samples.png")
    
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples = real_samples.to(device)
            real_samples_labels = torch.ones((batch_size, 1)).to(device)
            latent_space_samples = torch.randn((batch_size, 100)).to(device)
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1)).to(device)
            
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Training the generator
            latent_space_samples = torch.randn((batch_size, 100)).to(device)
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if n == batch_size - 1:
                print(f"Epoch: {epoch} | Loss D.: {loss_discriminator:.4f} | Loss G.: {loss_generator:.4f}")
        
        # Save generated samples every epoch
        latent_space_samples = torch.randn(batch_size, 100).to(device)
        generated_samples = generator(latent_space_samples)
        generated_samples = generated_samples.cpu().detach()
        
        save_path = f"{results_dir}/epoch_{epoch:03d}.png"
        plot_samples(generated_samples, f"Generated Samples - Epoch {epoch}", save_path)
    
    # Generate and plot final samples
    latent_space_samples = torch.randn(batch_size, 100).to(device)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.cpu().detach()
    
    plot_samples(generated_samples, "Generated Samples - Final", f"{results_dir}/final_samples.png")
    
    print(f"\nAll results saved to: {results_dir}")
    return discriminator, generator

if __name__ == "__main__":
    discriminator, generator = train_gan(num_epochs=50, batch_size=32, lr=0.0001)