from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import json

fig, ax = pyplot.subplots(1, 2)
LATENT_SPACE_IMAGE_FILE = 'latent_space_image.json'
TRAINED_MODEL_FILE = 'trained_model_cnn.h5'
EPOCHS = 10
BATCH_SIZE = 64

def LoadEncoder():
	""" Loads encoder from the file """

	full_model = tf.keras.models.load_model(TRAINED_MODEL_FILE)

	input_layer = full_model.layers[0]
	hidden_layer = full_model.layers[1:9]

	model = tf.keras.Sequential([input_layer] + hidden_layer) # add input and hidden layers
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

	print("Encoder summary: ", model.summary())

	return model

def LoadDecoder():
	""" Loads decoder from the file """

	full_model = tf.keras.models.load_model(TRAINED_MODEL_FILE)

	input_layer = full_model.layers[9]
	hidden_layer = full_model.layers[9:17]

	model = tf.keras.Sequential([input_layer] + hidden_layer) # add input and hidden layers
	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
	model.build((1, 2))

	print("Decoder summary: ", model.summary())

	return model

def MakeConvolutionalModel():
	""" Builds the convolutional neural network """

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
	])

	model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

	print(model.summary())

	return model

def TrainModel(model, save=False):
	""" Trains the model """

	model.fit(train_x, train_x, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(test_x, test_x))
	if save: model.save(TRAINED_MODEL_FILE)

	return model

def ShowImgs(imgs):
	""" Shows 9 images from imgs variable (debug only) """

	for i in range(9):  
		img = np.zeros((28, 28))
		for j in range(28):
			for k in range(28):
				img[j, k] = imgs[i, j, k, 0]

		pyplot.subplot(330 + 1 + i)
		pyplot.imshow(img, cmap=pyplot.get_cmap('gray'))
	pyplot.show()

def UpdateImage(event):
	""" Generates and draws new image every time we move mouse cursor """

	if event.xdata is None or event.ydata is None:
		return
	
	if event.inaxes != ax[0]:
		return

	x, y = float(event.xdata), float(event.ydata)
	ax[1].clear()  # clear previous image
	decoder_predictions = decoder.predict([[x, y]])

	ax[1].imshow(np.reshape(decoder_predictions, (28, 28)), cmap='gray')  # display the original image
	canvas.draw()  # redraw the canvas

def MapInputToLatentSpace():
	""" Maps the input space to the latent space of only two variables """

	encoder_predictions = encoder.predict(test_x)
	data = [[], [], [], [], [], [], [], [], [], []]
	
	for i in range(len(test_x)):
		data[test_y[i]].append([str(encoder_predictions[i, 0]), str(encoder_predictions[i, 1])]) # we save predicted data from encoder to the class of the current sample

	json_parsed = json.dumps(data)

	with open(LATENT_SPACE_IMAGE_FILE, "w") as outfile:
		outfile.write(json_parsed)

def VisualiseLatentSpace():
	""" Loads and draws image of latent space """

	f = open(LATENT_SPACE_IMAGE_FILE)

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
	""" Main function """

	print("Started!")

	(train_x, train_y), (test_x, test_y) = mnist.load_data()
	train_x = train_x / 255.0
	test_x = test_x / 255.0

	train_x_reshaped = train_x.reshape((len(train_x), 784))
	test_x_reshaped = test_x.reshape((len(test_x), 784))

	model = MakeConvolutionalModel() # COMMENT THIS IF MODEL IS ALREADY TRAINED AND YOU ONLY WANT VISUALIZATION
	model = TrainModel(model, save=True) # COMMENT THIS IF MODEL IS ALREADY TRAINED AND YOU ONLY WANT VISUALIZATION

	encoder = LoadEncoder() # COMMENT THIS IF MODEL IS ALREADY TRAINED AND YOU ONLY WANT VISUALIZATION
	decoder = LoadDecoder()

	MapInputToLatentSpace() # COMMENT THIS IF MODEL IS ALREADY TRAINED AND YOU ONLY WANT VISUALIZATION

	""" This part is only visualization, with already trained model """

	ax[1].imshow(np.random.rand(28, 28), cmap='gray') # show random noise for start

	canvas = fig.canvas
	canvas.mpl_connect('motion_notify_event', UpdateImage) # subscribe to the mouse movement event

	VisualiseLatentSpace() # draw image of latent space representation

	pyplot.show()





	
	
	
