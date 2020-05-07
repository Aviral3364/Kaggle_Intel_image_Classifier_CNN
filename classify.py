import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from tensorflow.keras.layers import BatchNormalization
import time

start_time = time.time()

train_datagen = ImageDataGenerator(rescale=1./255,
								   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
	                                              batch_size=32,
                                                  target_size=(150, 150),
                                                  class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                         target_size=(150, 150),
                                                         batch_size=32,
                                                         class_mode='categorical')

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform', 
	                    input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), strides = (1,1), padding = 'valid', activation='relu', kernel_initializer='glorot_uniform',))
model.add(layers.MaxPooling2D(pool_size=(2, 2), padding = 'valid'))
model.add(BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(400, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(128, activation = 'relu', kernel_initializer='glorot_uniform'))
model.add(layers.Dense(6, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(training_set, epochs = 20)

preds = model.evaluate(test_set)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

##### Now predicting some random images

#image_num = [*range(1,201)]
dic = {}

for img in range(50) :
	try :
		img_path = 'dataset/prediction/' + str(img) + '.jpg'
		test_image = image.load_img(img_path, target_size = (150, 150))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		x = x/255.0
		result = model.predict(test_image)
		print('\n',result)
		dic.update((str(img),np.argmax(result)))
	except :
		continue

df2 = pd.DataFrame([dic])
df2.to_csv('Predicted_classes.csv', index = False)

model.summary()
model.save('Intel_Image_Classifier', include_optimizer = True)

print("The training and test time of CNN is %s seconds ---" % (time.time() - start_time))
print("\n")
print("***********************************************************************************************************")
print("THANK YOU FOR AVAILING THIS SERVICE")
print("This CNN had been implemented by AVIRAL SINGH")
print("***********************************************************************************************************")