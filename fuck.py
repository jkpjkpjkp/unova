from PIL import Image
import numpy as np

image = Image.fromarray(np.zeros((100, 200)), mode='L')

# display(image)
image.show()