# Malaria-dectection
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
parasitized=os.path.join(r'D:\full images\cell_images\Parasitized')
uninfected=os.path.join(r'D:\full images\cell_images\Uninfected')
train_uninfected_images=os.listdir(uninfected)
print(train_uninfected_images[:5])
train_infected_images=os.listdir(parasitized)
print(train_infected_images[:5])

batch_size=128
train_datagen=ImageDataGenerator(rescale=1/255)
train_generator=train_datagen.flow_from_directory(r'D:\full images\cell_images', target_size=(100,100),batch_size=batch_size,classes=['parasitized','uninfected'],class_mode='binary')
target_size=(100,100)

model=tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(100,100,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation='sigmoid')
        ])
model.summary()

model.compile(loss='binary_crossentropy',optimizer=tf.keras.optimizers.RMSprop(),metrics=['acc'])
total_sample=train_generator.n
num_epochs=30
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),epochs=num_epochs,verbose=1)


from keras.models import model_from_json
from keras.models import load_model
model_json=model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("model1.h5")
print("saved model to disk")

'''classes=['infected','uninfected']
a=list(result[0]).index(max(list(result[0])))
print(classes[a])'''
