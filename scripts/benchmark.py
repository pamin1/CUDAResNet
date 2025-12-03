import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
from torchvision.datasets import CIFAR10
import os

sample_size = 50


def loadCifar():
    dataset = CIFAR10(root="./assets", train=False, download=True)

    os.makedirs("assets/cifar10/images", exist_ok=True)

    labels_list = []
    for i in range(min(1000, len(dataset))):
        img, label = dataset[i]
        img = img.resize((224, 224))
        img.save(f"assets/cifar10/images/image_{i:04d}.png")
        labels_list.append(label)

    with open("assets/cifar10/labels.txt", "w") as f:
        for label in labels_list:
            f.write(f"{label}\n")


def launchTorch(use_cuda):
    # Load pretrained ResNet-18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if use_cuda:
        model = model.cuda()
    model.eval()

    # ImageNet normalization
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load class labels
    pwd = os.getcwd()
    with open(f"{pwd}/assets/imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]

    print("Warming up PyTorch...")
    if use_cuda:
        dummy_input = torch.randn(1, 3, 224, 224).cuda()
    else:
        dummy_input = torch.randn(1, 3, 224, 224)

    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)

    if use_cuda:
        torch.cuda.synchronize()
    print("Warmup complete, starting benchmark...")

    # Process images
    inference_times = []
    for i in range(sample_size):
        img_path = f"{pwd}/assets/cifar10/images/image_{i:04d}.png"

        # Load and preprocess
        img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)
        if use_cuda:
            input_batch = input_batch.cuda()

        # Inference with timing
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            output = model(input_batch)

        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        # Get prediction
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.topk(probabilities, 1)

        predicted_class = classes[top_idx.item()]
        confidence = top_prob.item()
        duration_ms = (end - start) * 1000
        inference_times.append(duration_ms)
        img = f"image_{i:04d}"

        print(
            f"{img:<12} Class: {predicted_class:<30} "
            f"Confidence: {confidence:>8.4f}  "
            f"Time: {duration_ms:>7.2f} ms"
        )

    # Calculate and print summary statistics
    total_time = sum(inference_times)
    ms_per_img = total_time / sample_size
    img_per_s = 1000 * sample_size / total_time

    print(f"Total Time: {total_time:.2f} ms")
    print(f"Avg Time: {ms_per_img:.2f} ms/img")
    print(f"Avg Freq: {img_per_s:.2f} Hz")


def main():
    if not os.path.exists("assets/cifar10/images"):
        print("CIFAR-10 images not found. Downloading and extracting...")
        loadCifar()
        print("CIFAR-10 images ready!")
    else:
        print("CIFAR-10 images already exist. Skipping download.")

    print("CPU Stats:")
    launchTorch(False)
    print("\nGPU Stats:")
    launchTorch(True)


if __name__ == "__main__":
    main()
