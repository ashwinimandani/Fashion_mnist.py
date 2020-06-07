import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_lables)=fashion_mnist.load_data()
train_images=train_images/255.0
test_images=test_images/255.0

def build_model():
  model=keras.Sequential([keras.layers.Conv2D(filters=64,
                                              kernel_initializer="he_normal",
                                              kernel_size=(3,3),
                                              activation="relu",
                                              input_shape=(28,28,1)),
                          keras.layers.Conv2D(filters=32,
                                              kernel_size=(5,5),
                                              activation="relu",
                                              ),
                          keras.layers.Flatten(),
                          keras.layers.Dense(units=64,activation="relu",kernel_initializer="he_normal"),
                          keras.layers.Dense(units=10,activation="softmax")
                          ])
  model.compile(keras.optimizers.Adam(0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])

  return model
  
  model=build_model()
  model.summary()
  
  model.fit(train_images,train_labels,epochs=5,validation_split=0.001)
  y_pred=model.predict(test_images)
  
  for i in range(len(y_pred)):
  y_pred[i]=list(map(int,y_pred[i]))
loss,accuracy=model.evaluate(train_images,train_labels)
  
  
