from torchvision import transforms

def build_base_transform():
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.445],
                             std=[0.269])
    ])

    return transform
