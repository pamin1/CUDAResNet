import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import time
import numpy as np

BENCHMARK_SAMPLE_SIZE = 1000


def benchmark_pytorch_resnet18():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Load CIFAR-10 test set
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cifar_test = CIFAR10(root="./assets", train=False, download=True, transform=transform)

    # Preload images to GPU
    print(f"Loading {BENCHMARK_SAMPLE_SIZE} images to GPU...")
    images = []
    labels = []
    for i in range(BENCHMARK_SAMPLE_SIZE):
        img, label = cifar_test[i]
        images.append(img.unsqueeze(0).to(device))  # Add batch dimension
        labels.append(label)

    print(f"Loaded {len(images)} images to GPU")

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for i in range(10):
            _ = model(images[i % 10])

    # Synchronize before benchmark
    torch.cuda.synchronize()

    # Benchmark
    print("Running benchmark...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()

        for i in range(BENCHMARK_SAMPLE_SIZE):
            output = model(images[i])

        end_event.record()

    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / BENCHMARK_SAMPLE_SIZE
    throughput = (BENCHMARK_SAMPLE_SIZE * 1000.0) / total_time_ms

    print("\n=== PyTorch CUDA Benchmark Results ===")
    print(f"Total time: {total_time_ms:.2f} ms")
    print(f"Average per image: {avg_time_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} img/s")

    # Memory usage
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # Sample prediction
    print("\n=== Sample Predictions ===")
    cifar_classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    with torch.no_grad():
        for i in range(5):
            output = model(images[i])
            pred = output.argmax(dim=1).item()
            print(
                f"Image {i} - True: {cifar_classes[labels[i]]}, Predicted class: {pred}"
            )


if __name__ == "__main__":
    benchmark_pytorch_resnet18()
