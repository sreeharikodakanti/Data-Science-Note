import flask
from flask import request, jsonify

model = test()


app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)


@app.route('/',methods=['GET'])
def default():
    return 'hi'
	
#app.run()

@app.route('/predict',methods=['POST'])
def predict():
	from sklearn.externals import joblib
	model = joblib.load('hppredict25AUG.ml')
	price = model.predict([[1500,1,1,1,1]])
	return str(price)

@app.route('/predict/animals', methods = ['POST'])
def predictAnimal():
	from sklearn.externals import joblib
	model = joblib.load('good_model_cnn.h5')
	return model.predict(

app.run()

def test():
	from keras.preprocessing.image import ImageDataGenerator

	datagen = ImageDataGenerator(
			rotation_range=40,
			width_shift_range=0.2,
			height_shift_range=0.2,
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			fill_mode='nearest')

	from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

	img = load_img('train/cats/cat.0.jpg')  # this is a PIL image
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

	# the .flow() command below generates batches of randomly transformed images
	# and saves the results to the `preview/` directory
	i = 0
	for batch in datagen.flow(x, batch_size=1,
							save_to_dir='train/preview', save_prefix='cat', save_format='jpeg'):
		i += 1
		if i > 20:
			break  # otherwise the generator would loop indefinitely


	from keras.models import Sequential
	from keras.layers import Conv2D, MaxPooling2D
	from keras.layers import Activation, Dropout, Flatten, Dense

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(150, 150,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(64))  # fully connect ANN Network
	model.add(Activation('relu'))
	model.add(Dropout(0.5))  # Generalizing (avoiding overfitting)
	model.add(Dense(1))
	model.add(Activation('sigmoid')) 

	model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

	batch_size = 16

	# this is the augmentation configuration we will use for training
	train_datagen = ImageDataGenerator(
			rescale=1./255,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True)

	# this is the augmentation configuration we will use for testing:
	# only rescaling
	test_datagen = ImageDataGenerator(rescale=1./255)

	# this is a generator that will read pictures found in
	# subfolers of 'data/train', and indefinitely generate
	# batches of augmented image data
	train_generator = train_datagen.flow_from_directory(
			'train/cats_dogs/train',  # this is the target directory
			target_size=(150, 150),  # all images will be resized to 150x150
			batch_size=batch_size,
			class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

	# this is a similar generator, for validation data
	validation_generator = test_datagen.flow_from_directory(
			'train/cats_dogs/validation',
			target_size=(150, 150),
			batch_size=batch_size,
			class_mode='binary')
	return model