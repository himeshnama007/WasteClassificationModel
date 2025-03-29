import pickle
import tensorflow as tf
import numpy as np

# let's prepare the data and generate the data

from tensorflow.keras.preprocessing.image import ImageDataGenerator

gen_train = ImageDataGenerator(rescale = 1/255, shear_range = 0.2, zoom_range = 0.2, 
                               brightness_range = (0.1, 0.5), horizontal_flip=True)

train_data = gen_train.flow_from_directory("D:\\internship\\aiplane\\basic project\\waste classification model\\waste dataset",
                                           target_size = (224, 224), batch_size = 200, class_mode="categorical")
# let's create a model
# here i'm going to use VGG16 model's parameter to solve this problem

from tensorflow.keras.applications.vgg16 import VGG16

# here i'm going to take input shape, weights and bias from imagenet and include top False means
# i want to add input, flatten and output layer by my self

vgg16 = VGG16(input_shape = (224, 224, 3), weights = "imagenet", include_top = False)
# now vgg16 weights are already train so i don't want to train that weights again

# so let's make trainable = False

for layer in vgg16.layers:
  layer.trainable = False
  
# let's add flatten layer or let's connect VGG16 with our own flatten layer

from tensorflow.keras import layers

x = layers.Flatten()(vgg16.output)

# now let's add output layers or prediction layer

prediction = layers.Dense(units = 6, activation="softmax")(x)

# creating a model object

model = tf.keras.models.Model(inputs = vgg16.input, outputs=prediction)
model.summary()

# now let's compile the model

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics =["accuracy"])

result = model.fit(train_data, epochs = 10, steps_per_epoch=len(train_data))
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)


print("model training is done")

