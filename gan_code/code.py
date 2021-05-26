#!/usr/bin/env python3

import os
from os import listdir
from keras.preprocessing.image import img_to_array
from numpy import savez_compressed
import numpy as np
import cv2
from PIL import Image
import sys

from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from datetime import datetime
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Set current working dir to where this script resides
######################################################
script_folder_path = os.path.dirname( __file__ )
os.chdir(script_folder_path)
cwd = os.getcwd()
######################################################


# Get os compatible path: win, linux, macos
######################################################
import posixpath
import ntpath
import platform
def to_os_comp_path(path):
	new_path = ''
	if 'linux'.lower() in platform.system().lower():
		new_path = path.replace(ntpath.sep, posixpath.sep)
	elif 'win'.lower() in platform.system().lower():
		new_path = path.replace(posixpath.sep, ntpath.sep)
	elif 'dar'.lower() in platform.system().lower():
		new_path = path.replace(ntpath.sep, posixpath.sep)

	return new_path
######################################################


# Get image contours based on median pixel intensity
#####################################################
def get_auto_canny_threshold_values(image, sigma=0.33):
	v = np.median(image)

	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))

	return (lower, upper)
#####################################################


# Extracting contours after image is converted from RGB to greyscale
#####################################################
def extract_contours(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    (lower, upper) = get_auto_canny_threshold_values(image)
    edge = cv2.Canny(image, lower, upper)

    return edge
#####################################################


# Resize image
#####################################################
def resize_image(image, dim):
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized
#####################################################


# Load images(raw and contour) into memory
#####################################################
def load_images(path, size=(256,256)):
    img_list = list()
    contour_img_list = list()
    loaded_image_count = 0

    print(f'Loading images from: {path}')
    # enumerate filenames in directory, assume all are images
    file_names = listdir(path)
    file_names.sort()
    print('Read: %d images.' % len(file_names))
    for filename in file_names:
        
        image = cv2.imread(to_os_comp_path(f'{path}\\{filename}'))
        color_converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_countours = extract_contours(color_converted)
        image_countours_3d = cv2.cvtColor(image_countours, cv2.COLOR_GRAY2RGB)
        
        image1 = resize_image(color_converted, size)
        image_countours1 = resize_image(image_countours_3d, size)
        
        pil_image = Image.fromarray(image1)
        pil_contour_image = Image.fromarray(image_countours1)

        # Convert to numpy array
        pixels1 = img_to_array(pil_image)
        pixels2 = img_to_array(pil_contour_image)

        img_list.append(pixels1)
        contour_img_list.append(pixels2)
        loaded_image_count += 1
        if loaded_image_count % 50 == 0:
            print(f'Loading... Loaded {loaded_image_count}')
    print(f'Finished loading from: {path}, loaded: {loaded_image_count} images')

    return (img_list, contour_img_list)
#####################################################


# Compress images to numpy zip
#####################################################
def compress_to_zip(path_to_load, normal_image_zip_name, contour_image_zip_name):
    (normal_images, contour_images) = load_images(path_to_load)

    print('Normal image collection shape: ', np.shape(normal_images))
    print('Contour image collection shape: ', np.shape(contour_images))

    savez_compressed(normal_image_zip_name, normal_images)
    savez_compressed(contour_image_zip_name, contour_images)

    print('Saved dataset: ', normal_image_zip_name)
    print('Saved dataset: ', contour_image_zip_name)
#####################################################


# Get discriminator block
#####################################################
def discriminator_block(layer_in, n_filters):
    init = RandomNormal(stddev=0.02)
    
    discriminator_block = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    discriminator_block = BatchNormalization()(discriminator_block)
    discriminator_block = LeakyReLU(alpha=0.2)(discriminator_block)

    return discriminator_block
#####################################################


# Get discriminator
#####################################################
def get_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)

    src_image = Input(shape=image_shape)
    target_image = Input(shape=image_shape)
    merged = Concatenate()([src_image, target_image])

    layers = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    layers = LeakyReLU(alpha=0.2)(layers)

    layers = discriminator_block(layers, 128)
    layers = discriminator_block(layers, 256)
    layers = discriminator_block(layers, 512)

    layers = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(layers)
    layers = BatchNormalization()(layers)
    layers = LeakyReLU(alpha=0.2)(layers)

    layers = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(layers)
    patch_out = Activation('sigmoid')(layers)

    model = Model([src_image, target_image], patch_out)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

    return model
#####################################################


# Get encoder block
#####################################################
def encoder_block(input_layer, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    layers = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_layer)

    if batchnorm:
        layers = BatchNormalization()(layers, training=True)
    
    layers = LeakyReLU(alpha=0.2)(layers)

    return layers
#####################################################


# Get decoder block
#####################################################
def decoder_block(input_layer, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    layers = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(input_layer)
    layers = BatchNormalization()(layers, training=True)

    if dropout:
        layers = Dropout(0.5)(layers, training=True)
        
    layers = Concatenate()([layers, skip_in])
    layers = Activation('relu')(layers)

    return layers
#####################################################


# Get generator
#####################################################
def get_generator(image_shape=(256,256,3)):
    init = RandomNormal(stddev=0.02)

    input_image = Input(shape=image_shape)
    encoder_layers_1 = encoder_block(input_image, 64, batchnorm=False)
    encoder_layers_2 = encoder_block(encoder_layers_1, 128)
    encoder_layers_3 = encoder_block(encoder_layers_2, 256)
    encoder_layers_4 = encoder_block(encoder_layers_3, 512)
    encoder_layers_5 = encoder_block(encoder_layers_4, 512)
    encoder_layers_6 = encoder_block(encoder_layers_5, 512)
    encoder_layers_7 = encoder_block(encoder_layers_6, 512)

    bottleneck_layers = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(encoder_layers_7)
    bottleneck_layers = Activation('relu')(bottleneck_layers)

    discriminator_layers_1 = decoder_block(bottleneck_layers, encoder_layers_7, 512)
    discriminator_layers_2 = decoder_block(discriminator_layers_1, encoder_layers_6, 512)
    discriminator_layers_3 = decoder_block(discriminator_layers_2, encoder_layers_5, 512)
    discriminator_layers_4 = decoder_block(discriminator_layers_3, encoder_layers_4, 512, dropout=False)
    discriminator_layers_5 = decoder_block(discriminator_layers_4, encoder_layers_3, 256, dropout=False)
    discriminator_layers_6 = decoder_block(discriminator_layers_5, encoder_layers_2, 128, dropout=False)
    discriminator_layers_7 = decoder_block(discriminator_layers_6, encoder_layers_1, 64, dropout=False)

    x = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(discriminator_layers_7)
    out_image = Activation('tanh')(x)
    
    return Model(input_image, out_image)
#####################################################

# Get GAN
#####################################################
def get_gan(g_model, d_model, image_shape):
	
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    input_src = Input(shape=image_shape)
    gen_out = g_model(input_src)
    dis_out = d_model([input_src, gen_out])
    model = Model(input_src, [dis_out, gen_out])

    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[1,100])

    return model
#####################################################

# Get images(raw and contour)
#####################################################
def load_real_samples(filename, filename1):
	data = load(filename)
	data1 = load(filename1)
	X1, X2 = data['arr_0'], data1['arr_0']

	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X2, X1]
#####################################################

# Generate images with real(1) label
#####################################################
def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))

    return [X1, X2], y
#####################################################

# Generate images with fake(0) label
#####################################################
def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))

    return X, y
#####################################################

# Save model progress
#####################################################
def summarize_performance(step, g_model, timestamp_string):
	root_save_path = f'{cwd}\\{timestamp_string}'
	if not os.path.exists(root_save_path):
		os.makedirs(root_save_path)
	
	# Save model progress
	filename2 = to_os_comp_path(f'{root_save_path}\\model_%06d.h5' % (step+1))
	g_model.save(filename2)

	print('>Saved: %s' % filename2)
#####################################################

# Get timestamp in string
#####################################################
def get_timestamp_in_string():
	now = datetime.now() # current date and time
	date_time = now.strftime("%Y-%m-%dT%H:%M:%S")

	return date_time
#####################################################

# Plot metric values
#####################################################
def plot_metric_values(nsteps, d1_loss, d2_loss, g_loss, save_path):
    fig, (d_loss_plot, g_loss_plot) = plt.subplots(2, 1)
    fig.suptitle('Discriminator and generator losses')

    # Plotting D1 loss 
    d_loss_plot.plot(nsteps, d1_loss, label = "D1_loss")
    d_loss_plot.plot(nsteps, d2_loss, label = "D2_loss")
    d_loss_plot.set_ylabel('Discriminator loss')
    d_loss_plot.legend()

    # Plotting G loss
    g_loss_plot.plot(nsteps, g_loss, label = "G_loss")
    g_loss_plot.set_ylabel('Generator loss')
    g_loss_plot.legend()
    g_loss_plot.set_xlabel('# of steps(step size - 20)')

    # Create dir if no such already exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.set_size_inches(25, 20)
    #fig.savefig(f'{save_path}/{nsteps[-1]}.jpg', dpi=300)

    plt.show()
#####################################################

# Train models
#####################################################
def execute_training(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    # Settings
    n_patch = d_model.output_shape[1]
    trainA, _ = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs
    
    print(f'Main settings:')
    print(f'------------')
    print(f'# of epochs: {n_epochs}')
    print(f'# of steps: {n_steps}')
    print(f'------------')
    
    timestamp = get_timestamp_in_string()
    n_steps_arr = []
    d_loss1_arr = []
    d_loss2_arr = []
    g_loss_arr = []
    start = datetime.now()

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)

        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])

        # Aggregate metric values
        if (i+1) % 10 == 0:
            n_steps_arr.append(i)
            d_loss1_arr.append(d_loss1)
            d_loss2_arr.append(d_loss2)
            g_loss_arr.append(g_loss)

        # Plot metric values
        if ((i+1 > 0) and ((i+1) % (bat_per_epo * 5) == 0)):
            plot_metric_values(np.array(n_steps_arr), np.array(d_loss1_arr), np.array(d_loss2_arr), np.array(g_loss_arr), to_os_comp_path(f'{timestamp}\\plots\\loss'))

        # Print loss metrics
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))

        # Save model progress
        if (i+1) % (bat_per_epo * 50) == 0:
            summarize_performance(i, g_model, timestamp)
        current_time = datetime.now()

        elapsed = (current_time - start).seconds
        going_to_take_time_in_seconds = (elapsed * n_steps) / (i+1)
        should_end_at = start + timedelta(0, int(going_to_take_time_in_seconds))
        should_end_at_str_formated = should_end_at.strftime("%H:%M:%S")
        print('Ends at: [%s]' % should_end_at_str_formated, end="\r", flush=True)
#####################################################


######################################################
try:
	input_image_path = sys.argv[1]
    
    # Compress to numpy zip
	compress_to_zip(input_image_path, f'{cwd}\\images.npz', f'{cwd}\\contours.npz')
	
    # Load images
	dataset = load_real_samples(f'{cwd}\\images.npz', f'{cwd}\\contours.npz')
	print('Loaded', dataset[0].shape, dataset[1].shape)
	
	image_shape = dataset[0].shape[1:]
    # Discriminator and generator models
	discriminator_model = get_discriminator(image_shape)
	generator_model = get_generator(image_shape)

	# GAN model
	gan_model = get_gan(generator_model, discriminator_model, image_shape)

    # Train 
	execute_training(discriminator_model, generator_model, gan_model, dataset)
except Exception as e:
	print("Unexpected error:", str(e))
	input("Press any key!")

input("Press any key!")