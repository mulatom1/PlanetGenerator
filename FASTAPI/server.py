import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from datetime import date
import uuid

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('SVG')

import numpy as np

import os

from sklearn.preprocessing import LabelEncoder

from typing import Union

from fastapi import FastAPI
from fastapi.responses import FileResponse


from keras.models import load_model

import random as rnd


app = FastAPI()

generator1 = load_model('./generator_1.h5')
generator2 = load_model('./generator_2.h5')
generator3 = load_model('./generator_3.h5')
generator4 = load_model('./generator_4.h5')
generator5 = load_model('./generator_5.h5')
generator6 = load_model('./generator_6.h5')
generator7 = load_model('./generator_7.h5')

@app.get("/")
def read_root():
    return {"Hello": "from new planet"}


@app.get("/get_planet/")
def read_get_planet():
    num_imgs = 16
    latent_dim = 100
    seed = tf.random.normal([num_imgs, latent_dim])
    
    idxgen = rnd.randint(0, 6)
    if idxgen == 0: generated_images = generator1(seed)
    if idxgen == 1: generated_images = generator2(seed)
    if idxgen == 2: generated_images = generator3(seed)
    if idxgen == 3: generated_images = generator4(seed)
    if idxgen == 4: generated_images = generator5(seed)
    if idxgen == 5: generated_images = generator6(seed)
    if idxgen == 6: generated_images = generator7(seed)

    generated_images = (generated_images * 127.5) + 127.5
    generated_images.numpy()

    filename = date.today().strftime("%d/%m/%Y").replace("/", "") + "_" + str(uuid.uuid4()).replace("-", "_")
    
    fig = plt.figure(figsize=(5, 5))
    img = keras.utils.array_to_img(generated_images[0]) 
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('./results_img/' + filename + '.jpg', dpi=100)
    plt.close(fig)

    return FileResponse('./results_img/' + filename + '.jpg')


@app.get("/get_planet_256/")
def read_get_planet_256():
    num_imgs = 16
    latent_dim = 100
    seed = tf.random.normal([num_imgs, latent_dim])

    generated_images = generator256(seed)
    generated_images = (generated_images * 127.5) + 127.5
    generated_images.numpy()

    filename = date.today().strftime("%d/%m/%Y").replace("/", "") + "_" + str(uuid.uuid4()).replace("-", "_")
    
    fig = plt.figure(figsize=(2.7, 2.7))
    img = keras.utils.array_to_img(generated_images[0]) 
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('./results_img/' + filename + '.jpg', dpi=300)
    plt.close(fig)

    return FileResponse('./results_img/' + filename + '.jpg')


@app.get("/get_planet_512/")
def read_get_planet_512():
    num_imgs = 16
    latent_dim = 100
    seed = tf.random.normal([num_imgs, latent_dim])

    generated_images = generator512(seed)
    generated_images = (generated_images * 127.5) + 127.5
    generated_images.numpy()

    filename = date.today().strftime("%d/%m/%Y").replace("/", "") + "_" + str(uuid.uuid4()).replace("-", "_")
    
    fig = plt.figure(figsize=(5.6, 5.6))
    img = keras.utils.array_to_img(generated_images[0]) 
    plt.imshow(img)
    plt.axis('off')
    plt.savefig('./results_img/' + filename + '.jpg', dpi=300)
    plt.close(fig)

    return FileResponse('./results_img/' + filename + '.jpg')
