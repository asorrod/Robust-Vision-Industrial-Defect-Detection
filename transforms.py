from torchvision.transforms import v2

def build_base_preprocess():
    transform =[
        v2.Grayscale(num_output_channels=1),
        v2.Resize((224,224)),
        v2.ToTensor()
    ]
    
    return transform

def build_base_normalize():
    return v2.Normalize(mean=[0.445],std=[0.269])

def build_corruption_transforms(name, severity):

    if name == "gaussian_noise":
        sigma = severity * 0.05
        return v2.GaussianNoise(mean=0.0, sigma=sigma, clip=True)
    
    elif name == "gaussian_blur":
        kernel = 3 + severity*2
        return v2.GaussianBlur(kernel_size=kernel)
    
    elif name == "brightness":
        factor = 1 + severity*0.2
        return v2.ColorJitter(brightness=factor)
    
    elif name == "occlusion":
        scale_min = 0.02 * severity
        scale_max = 0.05 * severity

        return v2.RandomErasing(
            p=1.0,
            scale=(scale_min, scale_max),
            ratio=(0.3, 0.3),
            value=0
        )
    
    else: 
        raise ValueError("Unknown corruptions")
    
def build_transform(perturbations=None):
    if perturbations is None:
        perturbations = []
    
    transforms_list = []
    transforms_list += build_base_preprocess()
    transforms_list += perturbations
    transforms_list.append(build_base_normalize())

    return v2.Compose(transforms_list)


