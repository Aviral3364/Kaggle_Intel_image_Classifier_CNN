import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('Intel_Image_Classifier')

model.summary()
n = 0
dic = {0 : 'Buildings',
       1 : 'Forest',
       2 : 'Glacier',
       3 : 'Mountain',
       4 : 'Sea',
       5 : 'Street'}

image_name = list()
label = list()
image_type = list()
print('Predicting Images......')
for img in range(24333) :
	try :
		img_path = 'dataset/prediction/' + str(img) + '.jpg'
		test_image = image.load_img(img_path, target_size = (150, 150))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		test_image = test_image/255.0
		result = model.predict(test_image)
		q = np.argmax(result)
		n += 1
		if n%200 == 0 :
			print('Images Predicted',n)
		image_name.append(str(img))
		label.append(q)
		z = dic.get(q)
		image_type.append(z)
	except :
		continue

list_of_tuples = list(zip(image_name, label, image_type))
print('Total Images Predicted',n)  

df2 = pd.DataFrame(list_of_tuples, columns = ['Image_Name', 'Image_label','Image_Type'])
df2.to_csv('Predicted_classes.csv', index = True)
