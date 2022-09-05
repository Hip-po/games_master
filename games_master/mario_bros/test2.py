import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import ndimage
from matplotlib import animation
import numpy as np
import os

files = os.listdir("screenshot")

fig = plt.figure()
myimages = []


for i in files:
    frame = np.load(f"screenshot/{i}")
    print(frame)
    myimages.append([frame])
    plt.imshow(frame)
    plt.show()

my_anim = animation.ArtistAnimation(myimages)

f = 'animation.mp4'
writervideo = animation.FFMpegWriter(fps=50)
my_anim.save(f, writer=writervideo)
plt.show()
