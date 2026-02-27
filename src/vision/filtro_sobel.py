# src/vision/filtro_sobel.py
import torch
import torch.nn.functional as F

def apply_sobel_l2(image_tensor):
    """
    Applies the Sobel operator (Eucliedean or L2 Norm) to a medical image tensor.
    Input: image_tensor of dimensions (Batch, Channel, Height, Width)
    Output: tensor containing the gradient magnitude.
    """
    # Clinical dimensionality verification
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    # Spatial derivative kernels (float32 for numerical stability)
    kernel_x = torch.tensor([[ -1.,  0.,  1.],
                             [ -2.,  0.,  2.],
                             [ -1.,  0.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
                             
    kernel_y = torch.tensor([[ -1., -2., -1.],
                             [  0.,  0.,  0.],
                             [  1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)

    # Convolution
    G_x = F.conv2d(image_tensor, kernel_x, padding=1)
    G_y = F.conv2d(image_tensor, kernel_y, padding=1)

    # Gradient Magnitude (Euclidean Norm)
    magnitude = torch.sqrt(G_x**2 + G_y**2)
    return magnitude
 
def apply_sobel_l1(image_tensor):
    """
    Using Manhattan or L1 norm. 
    """
    # Clinical dimensionality verification
    if len(image_tensor.shape) == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)
        
    # Spatial derivative kernels (float32 for numerical stability)
    kernel_x = torch.tensor([[ -1.,  0.,  1.],
                             [ -2.,  0.,  2.],
                             [ -1.,  0.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)
                             
    kernel_y = torch.tensor([[ -1., -2., -1.],
                             [  0.,  0.,  0.],
                             [  1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3)

    # Convolution
    G_x = F.conv2d(image_tensor, kernel_x, padding=1)
    G_y = F.conv2d(image_tensor, kernel_y, padding=1)

    # Gradient Magnitude (Manhattan Norm)
    magnitude = torch.abs(G_x + G_y)
    return magnitude
