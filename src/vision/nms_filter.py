import torch
import torch.nn.functional as F
# Non-Maximum Suppression (NMS) is a method used in object detection to remove extra boxes that are detected around the same object. When an object is detected multiple times with different bounding boxes, NMS keeps the best one and removes the rest. This helps us to make sure each object is counted only once, improving the accuracy and clarity of the results.
def convolution(image_tensor):
    """
    Recycled function from filter_sobel, with a twist, I will transform the function
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
 
'''Next we will do a Non-Maximum Suppression logic, using the torch.roll function
This function its like moving some dimension of the tensor x places 
This created shifted copies of the original matrix
For example, defining dim=3 as the columns, horizontal, width or x axis, if you shift the tensor Right (shifts=1, dims=3), the pixel currently residing at index (i,j) in the new tensor is the Left Neighbor M(i,j−1) from the original tensor.
I won't use a single for loop, because this is slow, we need to use some vectorized operations, and making some boolean masks

'''
def NMS(image_tensor):
   # The continuous gradient angles (from angle_deg of the above function) must be mathematically quantized into four discrete bins (0∘,45∘,90∘,135∘) to map onto the 8-connected discrete spatial geometry of a pixel matrix. 
    
    magnitude_matrix = convolution(image_tensor)[2]
    degree_matrix = convolution(image_tensor)[3]

    # Boolean masks
    zero_degree_mask = (degree_matrix < 22.5) | (degree_matrix >= 157.5)
    degree_mask_45 = (degree_matrix >= 22.5) |(degree_matrix < 67.5)
    degree_mask_90 = (degree_matrix <= 67.5 ) | (degree_matrix > 112.5)
    degree_mask_135 = (degree_matrix <= 112.5) | (degree_matrix > 157.5)
    
    # Calculating neighbors T - Top, B - Bottom, R - Right, L - Left
    Neighbor_T = torch.roll(magnitude_matrix, shifts = 1, dims = 2)
    Neighbor_B = torch.roll(magnitude_matrix, shifts = -1, dims = 2)
    Neighbor_R = torch.roll(magnitude_matrix, shifts = 1, dims = 3)
    Neighbor_L = torch.roll(magnitude_matrix, shifts = -1, dims = 3)
    Neighbor_TR = torch.roll(magnitude_matrix, shifts=(-1, 1), dims=(2, 3))
    Neighbor_BL = torch.roll(magnitude_matrix, shifts=(1, -1), dims=(2, 3))
    Neighbor_TL = torch.roll(magnitude_matrix, shifts=(-1, -1), dims=(2, 3))
    Neighbor_BR = torch.roll(magnitude_matrix, shifts=(1, 1), dims=(2, 3))
    
    # Aplly the non maxium suppresion in each neigbor
    Survivor_0 = (magnitude_matrix >= Neighbor_R) & (magnitude_matrix >= Neighbor_L)
    Survivor_45 = (magnitude_matrix >= Neighbor_TR) & (magnitude_matrix >= Neighbor_TL)
    Survivor_90 = (magnitude_matrix >= Neighbor_T) & (magnitude_matrix >= Neighbor_B)
    Survivor_135 = (magnitude_matrix >= Neighbor_TL) & (magnitude_matrix >= Neighbor_BR)

    # Compute the isolated 45-degree edges
    RESULT_0 = magnitude_matrix * zero_degree_mask * Survivor_0
    RESULT_45 = magnitude_matrix * degree_mask_45 * Survivor_45
    RESULT_90 = magnitude_matrix * degree_mask_90 * Survivor_90
    RESULT_135 = magnitude_matrix * degree_mask_135 * Survivor_135
    Final = RESULT_0 + RESULT_45 + RESULT_90 + RESULT_135
    
    return Final

def noise_cleaner(image_tensor):

    image = NMS(image_tensor)
    max_matrix = torch.max(image)
    #We assing two treshold for cleaning the noise 
    t_high = max_matrix * 0.15   # 15% of max intensity
    t_low = max_matrix * 0.05   # 5% of max intensity

    # Boolean Classification Matrices
    strong_edge = (image >= t_high)
    weak_edge  = (image >= t_low) & (image < t_high)

    # Output Matrix Construction
    # We assign arbitrary distinct values so we can track them in the next step.
    # E.g., Strong = 255 (White), Weak = 50 (Dark Gray), Noise = 0 (Black)
    treshold_matrix = (strong_edge * 255) + (weak_edge * 50)



    return treshold_matrix
