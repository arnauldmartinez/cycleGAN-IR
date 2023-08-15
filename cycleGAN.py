import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import random
import time
from IPython.display import clear_output

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
EPOCHS = 50

# Converts image to tensor
def parse_image_RGB(filename):
  parts = tf.strings.split(filename, os.sep)
  label = 'RGB'

  # data from an image file is stored in image
  image = tf.io.read_file(filename)
  # the image is decoded using decode_jpeg, but for other extensions you can also use decode_image
  image = tf.image.decode_jpeg(image, channels=3)
  # converting the datatype from long to float (here the data is already normalized so you cannot convert it to integers)
  image = tf.image.convert_image_dtype(image, tf.float32)
  # resizing the image to our desided size
  image = tf.image.resize(image, [256, 256])
  return image, label

# Converts image to tensor
def parse_image_IR(filename):
  parts = tf.strings.split(filename, os.sep)
  label = 'IR'

  # data from an image file is stored in image
  image = tf.io.read_file(filename)
  # the image is decoded using decode_jpeg, but for other extensions you can also use decode_image
  image = tf.image.decode_jpeg(image, channels=3)
  # converting the datatype from long to float (here the data is already normalized so you cannot convert it to integers)
  image = tf.image.convert_image_dtype(image, tf.float32)
  # resizing the image to our desided size
  image = tf.image.resize(image, [256, 256])
  return image, label

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image

# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label):
  image = random_jitter(image)
  # image = normalize(image)
  return image

def preprocess_image_test(image, label):
  image = normalize(image)
  return image

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

def generate_images(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
  
  return (total_gen_g_loss, total_gen_f_loss)
                                

def generate_loss_graph(x_coords, y_coords):
    plt.plot(x_coords, y_coords)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

training_RGB = tf.data.Dataset.list_files("/Users/arnauldmartinez/Desktop/CycleGAN Example/NewColor2Infrared/Training/RGB/*.jpg", shuffle=True)
training_IR = tf.data.Dataset.list_files("/Users/arnauldmartinez/Desktop/CycleGAN Example/NewColor2Infrared/Training/IR/*.jpg", shuffle=True)

testing_RGB = tf.data.Dataset.list_files("/Users/arnauldmartinez/Desktop/CycleGAN Example/NewColor2Infrared/Testing/RGB/*.jpg", shuffle=True)

training_RGB = training_RGB.map(parse_image_RGB)
training_IR = training_IR.map(parse_image_IR)
testing_RGB = testing_RGB.map(parse_image_RGB)

training_RGB = training_RGB.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

training_IR = training_IR.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

testing_RGB = testing_RGB.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

sample_RGB = next(iter(training_RGB))
sample_IR = next(iter(training_IR))

plt.subplot(121)
plt.title('RGB')
plt.imshow(sample_RGB[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('RGB with random jitter')
plt.imshow(random_jitter(sample_RGB[0]) * 0.5 + 0.5)

plt.show()

plt.subplot(121)
plt.title('IR')
plt.imshow(sample_IR[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('IR with random jitter')
plt.imshow(random_jitter(sample_IR[0]) * 0.5 + 0.5)

plt.show()

OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_IR = generator_g(sample_RGB)
to_RGB = generator_f(sample_IR)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_RGB, to_IR, sample_IR, to_RGB]
title = ['RGB', 'To IR', 'IR', 'To RGB']

for i in range(len(imgs)):
    plt.subplot(2, 2, i+1)
    plt.title(title[i])
    if i % 2 == 0:
        plt.imshow(imgs[i][0] * 0.5 + 0.5)
    else:
        plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a valid IR image?')
plt.imshow(discriminator_y(sample_IR)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a valid RGB image?')
plt.imshow(discriminator_x(sample_RGB)[0, ..., -1], cmap='RdBu_r')

plt.show()

generator_g_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.legacy.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                        generator_f=generator_f,
                        discriminator_x=discriminator_x,
                        discriminator_y=discriminator_y,
                        generator_g_optimizer=generator_g_optimizer,
                        generator_f_optimizer=generator_f_optimizer,
                        discriminator_x_optimizer=discriminator_x_optimizer,
                        discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

gen_g_loss = []
gen_f_loss = []
t = []

for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  epoch_loss_g = 0
  epoch_loss_f = 0
  for image_x, image_y in tf.data.Dataset.zip((training_RGB, training_IR)):
    g_loss, f_loss = train_step(image_x, image_y)
    g_loss = g_loss.numpy()
    f_loss = f_loss.numpy()
    epoch_loss_g += g_loss
    epoch_loss_f += f_loss
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    # TODO: NOTE: Uncomment
    generate_images(generator_g, sample_RGB)

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
  gen_g_loss.append(epoch_loss_g / n)
  gen_f_loss.append(epoch_loss_f / n)
  t.append(epoch)

  print(str(gen_g_loss))
  print(str(gen_f_loss))
  print(str(t))
  if (epoch + 1) % 5 == 0:
    generate_loss_graph(t, gen_g_loss)

n = 1
for inp in testing_RGB.take(5):
  prediction = generator_g(inp)

  IR_image_name = "/Users/arnauldmartinez/Desktop/CycleGAN Example/output/IRFigure" + str(n) + ".png"
  RGB_image_name = "/Users/arnauldmartinez/Desktop/CycleGAN Example/output/RGBFigure" + str(n) + ".png"
  tf.keras.preprocessing.image.save_img(IR_image_name, prediction[0])
  tf.keras.preprocessing.image.save_img(RGB_image_name, inp[0])
  n += 1

num_images = 5

for p in range(1, num_images+1):
    image_name = "output/IRFigure" + str(p) + ".png"
    input_image = Image.open(image_name)
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

    output_name = "output/IRFigure" + str(p) + "HotPixel.png"
    input_image.save(output_name, format="png")

print('Bruh')