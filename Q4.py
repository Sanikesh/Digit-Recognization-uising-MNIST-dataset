import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#This code is to train the model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

model.save('digits.model')

# Once the model is trained, it is not required to train it again. We can load the saved model.
#
# model=tf.keras.models.load_model('digits.model')


path='Digit/Eight.jpg'
img=cv2.imread(path)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:, :, 0]
s = hsv[:, :, 1]
v = hsv[:, :, 2]

blur = cv2.blur(v, (15, 15))

dnoise = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)
ret, binary = cv2.threshold(dnoise, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)

eros = cv2.erode(binary, kernel, iterations=1)
dial = cv2.dilate(binary, kernel, iterations=1)
imgd=np.invert(dial)
img2=cv2.resize(dial,(28,28))

prediction=model.predict(np.array([img2]))
print('This digit predicted is:',np.argmax(prediction))

cv2.imshow('Original Image',img)
cv2.imshow('Processed Image',imgd)
cv2.waitKey(0)

# fig, ax = plt.subplots(2, 4, figsize=(15, 7))
# ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# ax[0, 0].set_title('Original')
# ax[0, 1].imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
# ax[0, 1].set_title('HSV')
# ax[0, 2].imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
# ax[0, 2].set_title('Value')
# ax[0, 3].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
# ax[0, 3].set_title('Blur')
# ax[1, 0].imshow(cv2.cvtColor(dnoise, cv2.COLOR_BGR2RGB))
# ax[1, 0].set_title('DeNoised')
# ax[1, 1].imshow(binary, cmap='gray')
# ax[1, 1].set_title('Binary')
# ax[1, 2].imshow(eros, cmap='gray')
# ax[1, 2].set_title("Eroded")
# ax[1, 3].imshow(dial, cmap='gray')
# ax[1, 3].set_title('Dialted')
# fig.suptitle('Graph', fontweight="bold")