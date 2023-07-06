from keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten,BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
#loads fashion mnist dataset using module fashion_mnist

(trainX, trainy), (testX, testy) = fashion_mnist.load_data()  

#Display the  number of samples and dimension of each image in the dataset

print('Train: X = ', trainX.shape)                    
print('Test: X = ', testX.shape)                      
for i in range(1, 9):
    plt.subplot(4, 4, i)
    plt.imshow(trainX[i])
plt.show()

#The additional dimension added representing channel(it becomes (batch_size, height, width, channels=1))
trainX = np.expand_dims(trainX, -1)      
testX = np.expand_dims(testX, -1)

print(trainX.shape)
model = Sequential([
    Conv2D(64, (5, 5),padding="same",activation="relu",input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(),
    Conv2D(128, (5, 5), padding="same",activation="relu"),
    MaxPooling2D(),
    Conv2D(256, (5, 5), padding="same",activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(10, activation="softmax")
    ])
model.compile(optimizer=Adam(learning_rate=1e-3),    
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

#summary of model architecture, displaying the layers, output shape, and the number of trainable parameters
model.summary()          

#training history is stored in history variable
#takes preprocessed training data (trainX) and corresponding labels (trainy) as inputs
#10 epochs with 100 steps per epoch
#parameter splits portion (33%) of training data for validation during training
history = model.fit(                
    trainX.astype(np.float32), trainy.astype(np.float32),  
    epochs=10,                                              
    steps_per_epoch=100,
    validation_split=0.33                         
)
score = model.evaluate(trainX, trainy, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#saves the trained model weights to a file named 'model.h5' in the current directory
model.save_weights('./model.h5', overwrite=True)   

#plot the training and validation accuracies from the training history
#creates a line plot showing the change in accuracy over epochs
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()                                              

#plot the training and validation losses from the training history
#creates a line plot showing the change in loss over epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()                                                

#labels for different classes in Fashion MNIST dataset
#predictions made using activation functions and index of highest predicted value is extracted using argmax()
labels = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat',   
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']    
predictions = model.predict(testX)              
for i in range(10):
    print("Prediction "+ str(i)+":"+ labels[(np.argmax(np.round(predictions[i])))])
    print("Actual " + str(i) + ":" + labels[(testy[i])])
    plt.imshow(testX[i], cmap=plt.get_cmap('gray'))  
    plt.show()