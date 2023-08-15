import random
from PIL import Image
input_image = Image.open("output/Figure0.png")
pixel_map = input_image.load()
  
# Extracting the width and height
# of the image:
width, height = input_image.size
buffer = 10
num_hot_pixels = 10

for i in range(num_hot_pixels):
  rand_x = random.randint(buffer, width-buffer)
  rand_y = random.randint(buffer, height-buffer)

  for x in range(rand_x - 2, rand_x + 2):
    for y in range(rand_y - 2, rand_y + 2):
      pixel_map[x, y] = (255, 255, 255)

input_image.save("output/FigureModified.png", format="png")