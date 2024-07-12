Food Item Recognition and Calorie Estimation

Project Overview

This project involves developing a model to accurately recognize food items from images and estimate their calorie content. The objective is to enable users to track their dietary intake and make informed food choices.

Dataset

The Food-101 dataset from Kaggle is used for training and evaluation.

Model Architectures

Three different pre-trained deep learning architectures were utilized:

DenseNet201
VGG19
InceptionV3
Code Description

Imports
python
Copy code
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, AveragePooling2D
from keras.regularizers import l2
import keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
DenseNet201 Model
python
Copy code
new_input = Input(shape=(224, 224, 3))
base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=new_input)
for layer in base_model.layers[:]:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(101, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
results = model.fit(train_data, epochs=50, validation_data=test_data,
                    steps_per_epoch=len(train_data), validation_steps=len(test_data),
                    callbacks=EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True))
VGG19 Model
python
Copy code
from tensorflow.keras.applications import VGG19
vgg = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg.layers:
    layer.trainable = False

model2 = Sequential()
model2.add(vgg)
model2.add(Flatten())
model2.add(Dense(4096, activation='relu'))
model2.add(Dense(4096, activation='relu'))
model2.add(Dropout(0.25))
model2.add(Dense(101, activation='softmax'))
model2.summary()

model2.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
results2 = model2.fit(train_data, epochs=50, validation_data=test_data,
                      steps_per_epoch=len(train_data), validation_steps=len(test_data),
                      callbacks=EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True))
InceptionV3 Model
python
Copy code
from keras.applications.inception_v3 import InceptionV3
base_model3 = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

x = base_model3.output
x = AveragePooling2D()(x)
x = Dropout(.5)(x)
x = Flatten()(x)
x = Dense(101, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)
model3 = Model(inputs=base_model3.input, outputs=x)
model3.summary()

opt = SGD(lr=.1, momentum=.9)
model3.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

results3 = model3.fit(train_data, epochs=10, validation_data=test_data,
                      steps_per_epoch=len(train_data), validation_steps=len(test_data),
                      callbacks=EarlyStopping(patience=2, monitor='val_accuracy', restore_best_weights=True))

model3.save("model_food_1012.h5")
Model Evaluation and Visualization
python
Copy code
loss, acc = model3.evaluate(test_data)
print("Test Accuracy:", round(acc*100, 2), "%", "\nTest Loss:", round(loss, 4))

fig = plt.figure()
plt.plot(results3['accuracy'], c='blue', label='accuracy')
plt.plot(results3['loss'], c='red', label='loss')
plt.title('Training data')
plt.legend(loc='upper right')
plt.show()

fig = plt.figure()
plt.plot(results3.history['val_accuracy'], c='blue', label='val accuracy')
plt.plot(results3.history['val_loss'], c='red', label='val loss')
plt.title('Testing data')
plt.legend(loc='upper right')
plt.show()

# Heatmap of Confusion Matrix
yp = model3.predict(test_data).argmax(axis=1).reshape(-1,)
m = pd.crosstab(test_data.labels, yp, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize=(30, 30))
sn.heatmap(m, annot=True, cmap='Blues')
plt.show()
Sample Predictions

python
Copy code
print("Sample Predictions")

macarons = load_img("/kaggle/input/food-101/food-101/food-101/images/macarons/2428554.jpg", target_size=(224, 224))
pizza = load_img("/kaggle/input/food-101/food-101/food-101/images/pizza/768276.jpg", target_size=(224, 224, 3))
donuts = load_img("/kaggle/input/food-101/food-101/food-101/images/donuts/2563686.jpg", target_size=(224, 224, 3))
toast = load_img("/kaggle/input/food-101/food-101/food-101/images/french_toast/2769309.jpg", target_size=(224, 224, 3))
fries = load_img("/kaggle/input/food-101/food-101/food-101/images/french_fries/2246621.jpg", target_size=(224, 224))
ice = load_img("/kaggle/input/food-101/food-101/food-101/images/ice_cream/579407.jpg", target_size=(224, 224))

# Process and Predict
samples = [macarons, fries, ice, pizza, donuts, toast]
samples = [img_to_array(sample)/255 for sample in samples]
samples = [sample.reshape(1, 224, 224, 3) for sample in samples]

for i, sample in enumerate(samples):
    p = model3.predict(sample).argmax()
    print(f"Class {p}: {values[p]}")
    print(f"Calories: {calories[p]}\nNote: {s}")
Acknowledgements

This project was completed as part of my Machine Learning internship at Prodigy InfoTech. A big thank you to Prodigy InfoTech for this incredible opportunity!

License

This project is licensed under the MIT License - see the LICENSE.md file for details.

Contact

Santhosh Reddy
GitHub
LinkedIn

