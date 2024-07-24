import matplotlib.pyplot as plt
import numpy as np




def imshow(img):
    """
    Display the input images in a plot.
    
    Parameters:
    img (torch.Tensor): The grid of images to display.
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()