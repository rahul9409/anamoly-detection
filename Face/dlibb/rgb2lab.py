#!/usr/local/bin/python3
import numpy as np
from PIL import Image, ImageCms

# Open image and discard alpha channel which makes wheel round rather than square
im = Image.open('images/3.jpg').convert('RGB')
# Convert to Lab colourspace
srgb_p = ImageCms.createProfile("sRGB")
lab_p  = ImageCms.createProfile("LAB")

rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
Lab = ImageCms.applyTransform(im, rgb2lab)

# Split into constituent channels so we can save 3 separate greyscales
L, a, b = Lab.split()
#a_np[1] is the a part of LAB
a_np = np.array(a)
print(a)
print(a_np[1])

L.save('L.png')
a.save('a.png')
b.save('b.png')
