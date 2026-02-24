import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

imagen = torch.zeros(1, 1, 100, 100) 
imagen[0, 0, 30:70, 30:70] = 1.0 

kernel_sobel = torch.tensor([
    [-1., -2., -1.],
    [0., 0., 0.],
    [1., 2., 1.]
]).view(1, 1, 3, 3) # Redimensionamos a (Canales_Salida, Canales_Entrada, Alto, Ancho)
kernel_sobel2 = torch.tensor([
    [-1., 0., 1.],
    [-1., 0., 2.],
    [-1., 0., 1.]
]).view(1, 1, 3, 3)
# Convolution
resultado = F.conv2d(imagen, kernel_sobel, padding=1)
resultado2 = F.conv2d(imagen, kernel_sobel2, padding=1)


plt.figure(figsize=(10, 5))
resultado_final = resultado + resultado2
plt.subplot(1, 2, 1)
plt.title("Origninal image (El Cuadrado)")
plt.imshow(imagen[0, 0], cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Lo que ve la Matemática (Bordes Verticales)")
plt.imshow(resultado[0, 0], cmap='gray')

plt.show()
