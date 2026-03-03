import torch
import torch.nn.functional as F

def canny(image_tensor):
    """
    Recycled function from filter_sobel, but with a twist, I will transform the function
    into a canny filter, using the arctan2 funcion.
    The arctan2 function has as input two vectors, and the output is the angle between them
    in Radians, you could investigate more in deep, but it derives from linear algebra and the 
    dot product in the space
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
    angle_rad = torch.atan2(G_y, G_x)
    angle_deg = angle_rad * (180.0 / torch.pi)
    angle_deg = angle_deg % 180.0
    
    # Output the complete spatial derivative state
    return G_x, G_y, magnitude, angle_deg
    return magnitude
 
