from keras.datasets import mnist
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot
import json

fig, ax = pyplot.subplots(1, 2)

def MapRange(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax
                  - outMin))

def LoadEncoder():
	full_model = tf.keras.models.load_model('trained_model_cnn_1.h5')

	input_layer = full_model.layers[0]
	hidden_layer = full_model.layers[1:9]

	model = tf.keras.Sequential([input_layer] + hidden_layer) # ADD INPUT AND HIDDEN LAYERS
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

	print("Encoder summary: ", model.summary())
	# input_layer = full_model.layers[0]
	# hidden_layer = full_model.layers[1]

	# model = tf.keras.Sequential([input_layer] + [hidden_layer]) # ADD INPUT AND HIDDEN LAYERS
	# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

	return model

def LoadDecoder():
	full_model = tf.keras.models.load_model('trained_model_cnn_1.h5')

	input_layer = full_model.layers[9]
	hidden_layer = full_model.layers[9:17]

	model = tf.keras.Sequential([input_layer] + hidden_layer) # ADD INPUT AND HIDDEN LAYERS
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
	model.build((1, 2))

	print("Decoder summary: ", model.summary())
	# input_layer = full_model.layers[2]
	# hidden_layer = full_model.layers[3]

	# model = tf.keras.Sequential([input_layer] + [hidden_layer]) # ADD INPUT AND HIDDEN LAYERS
	# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

	return model

encoder = LoadEncoder()
decoder = LoadDecoder()

def LoadFullModel():
	model = tf.keras.models.load_model('trained_model.h5')
	return model

def MakeConvolutionalModel():
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(28, 28, 1)),
		tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=(28, 28, 1)),		
		tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="valid", activation="relu", input_shape=(28, 28, 1)),		
		tf.keras.layers.MaxPool2D((2, 2)),																				
		tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu"),	
		tf.keras.layers.Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu"),			
		tf.keras.layers.MaxPool2D((2, 2)),																				
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dropout(0.5),

		tf.keras.layers.Dense(2, activation=None),
		tf.keras.layers.Dense(1024, activation=None),
		tf.keras.layers.Reshape((4, 4, 64)),
		tf.keras.layers.UpSampling2D((2, 2)),
		tf.keras.layers.Conv2DTranspose(64, 3, padding="valid", activation="relu"),
		tf.keras.layers.Conv2DTranspose(64, 3, padding="valid", activation="relu"),
		tf.keras.layers.UpSampling2D((2, 2)),
		tf.keras.layers.Conv2DTranspose(32, 3, padding="valid", activation="relu"),
		tf.keras.layers.Conv2DTranspose(1, 3, padding="valid", activation="relu"),

		#tf.keras.layers.Dense(2, activation="tanh"),
		#tf.keras.layers.Dense(49, activation="tanh"),
		#tf.keras.layers.Dense(196, activation="tanh"),
		#tf.keras.layers.Dense(784, activation="sigmoid"),


		# OLD 1
		# tf.keras.layers.Input(shape=(28, 28, 1)),
		# tf.keras.layers.Conv2D(32, 3, strides=(1, 1), padding="same", activation="relu", input_shape=(28, 28, 1)),		# from 28x28 to 28x28
		# tf.keras.layers.MaxPool2D((2, 2)),																				# from 28x28 to 14x14
		# tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),												# form 14x14 to 14x14
		# tf.keras.layers.MaxPool2D((2, 2)),																				# from 7x7 to 3x3 ???
		# tf.keras.layers.Flatten(),
		# tf.keras.layers.Dropout(0.5),

		# tf.keras.layers.Dense(512, activation="relu"), # code layer

		# tf.keras.layers.Dense(10, activation="relu"),
		# tf.keras.layers.Dropout(0.5),
		# tf.keras.layers.Dense(256, activation="relu"),
		# tf.keras.layers.Dropout(0.5),
		# tf.keras.layers.Dense(512, activation="relu"),
		# tf.keras.layers.Dropout(0.5),
		# tf.keras.layers.Dense(784, activation="sigmoid"),
	])

	model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

	print(model.summary())

	return model

def MakeModel():
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(784,)),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(32, activation='relu'),
		tf.keras.layers.Dense(16, activation='relu'),
		tf.keras.layers.Dense(8, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(4, activation='relu'),
		tf.keras.layers.Dense(2, activation='relu'),
		tf.keras.layers.Dense(4, activation='relu'),
		tf.keras.layers.Dense(8, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(16, activation='relu'),
		tf.keras.layers.Dense(32, activation='relu'),
		tf.keras.layers.Dense(64, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(256, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(512, activation='relu'),
		tf.keras.layers.Dense(784, activation='sigmoid')
	])

	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

	return model

def TrainModel():
	model = MakeConvolutionalModel()

	model.fit(train_x, train_x, epochs=20, batch_size=32)

	predictions = model.predict(train_x)

	#model.save('trained_model_cnn_2.h5')

	return predictions

def ShowImgs(imgs):
	for i in range(9):  
		img = np.zeros((28, 28))
		for j in range(28):
			for k in range(28):
				img[j, k] = imgs[i, j, k, 0]

		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(img, cmap=pyplot.get_cmap('gray'))
	pyplot.show()

def update_image(event):
	if event.xdata is None or event.ydata is None:
		return
	
	if event.inaxes != ax[0]:
		return

	x, y = float(event.xdata), float(event.ydata)
	#print(x, " : ", y)
	#if 0 <= x < 28 and 0 <= y < 28:
	ax[1].clear()  # Clear the previous image
	decoder_predictions = decoder.predict([[x, y]])

	ax[1].imshow(np.reshape(decoder_predictions, (28, 28)), cmap='gray')  # Display the original image
	canvas.draw()  # Redraw the canvas

def MapInputToLatentSpace():
	encoder_predictions = encoder.predict(test_x)
	data = [[], [], [], [], [], [], [], [], [], []]
	
	for i in range(len(test_x)):
		data[test_y[i]].append([str(encoder_predictions[i, 0]), str(encoder_predictions[i, 1])]) # WE SAVE PREDICTED DATA FROM ENCODER TO THE CLASS OF THE CURRENT SAMPLE

	json_parsed = json.dumps(data)
	print("JSON: ", json_parsed)

	with open("latent_space_image_2.json", "w") as outfile:
		outfile.write(json_parsed)

def VisualiseLatentSpace():
	f = open('latent_space_image.json')

	data = json.load(f)
	colors = ["red", "green", "blue", "yellow", "black", "orange", "pink", "brown", "cyan", "gray"]

	for i in range(10):
		plot_x = []
		plot_y = []
		for j in range(len(data[i])):
			plot_x.append(float(data[i][j][0]))
			plot_y.append(float(data[i][j][1]))
		ax[0].scatter(plot_x, plot_y, c=colors[i])


if __name__ == '__main__':
	print("Start")

	(train_x, train_y), (test_x, test_y) = mnist.load_data()
	train_x = train_x / 255.0
	test_x = test_x / 255.0

	#for i in range(9):  
	#	pyplot.subplot(330 + 1 + i)
	#	pyplot.imshow(train_x[i], cmap=pyplot.get_cmap('gray'))
	#pyplot.show()

	train_x_reshaped = train_x.reshape((len(train_x), 784))
	test_x_reshaped = test_x.reshape((len(test_x), 784))

	#predictions = TrainModel()
	#ShowImgs(predictions)

	#encoder_predictions = encoder.predict(train_x)
	#print("Encoder predictions: ", encoder_predictions)
	
	#MapInputToLatentSpace()

	ax[1].imshow(np.random.rand(28, 28), cmap='gray')

	canvas = fig.canvas
	canvas.mpl_connect('motion_notify_event', update_image)

	VisualiseLatentSpace()

	pyplot.show()

	# decoder_predictions = decoder.predict(encoder_predictions)
	# print("Decoder predictions shape: ", decoder_predictions.shape)
	# print("Decoder predictions: ", decoder_predictions)
	# for i in range(9):  
	# 	pyplot.subplot(330 + 1 + i)
	# 	pyplot.imshow(np.reshape(decoder_predictions[i], (28, 28)), cmap=pyplot.get_cmap('gray'))
	# pyplot.show()

	# ************************** VISUALISE ****************************

	# image = np.random.rand(28, 28)
	# ax.imshow(image, cmap='gray')

	# canvas = fig.canvas
	# canvas.mpl_connect('motion_notify_event', update_image)

	# pyplot.show()



	
	
	