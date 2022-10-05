import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras_vggface.vggface import VGGFace

names=[]
with open('classes.txt') as f:
    while True:
        line = f.readline()
        if not line:
            break
        name=line.strip()
        names.append(name)
print(names)
num_classes=len(names)

def count_images(dir):
    cnt=0
    for folder in os.listdir(dir):
        for image in os.listdir(dir+'/'+folder):
            if os.path.isfile(os.path.join(dir, folder,image)):
                cnt+=1
    return cnt
train_len=count_images('images/train_cropped')
test_len=count_images('images/test_cropped')


trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="images/train_cropped",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="images/test_cropped", target_size=(224,224))


vgg16_model = VGGFace(include_top=False,
model='vgg16',
input_shape=(224, 224, 3))
print("num_of_layers",len(vgg16_model.layers))

my_model=Sequential()
my_model.add(vgg16_model)
my_model.add(GlobalAveragePooling2D())
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dense(1024, activation='relu'))
my_model.add(Dense(512, activation='relu'))
my_model.add(Dense(num_classes, activation='softmax'))
my_model.layers[0].trainable=False
print("num_of_layers",len(my_model.layers))
my_model.summary()


opt = Adam(lr=0.01)
my_model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
my_model.summary()

steps_per_epoch =train_len//32
validation_steps = test_len//32
checkpoint = ModelCheckpoint("models/vgg16_1.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=3, verbose=1, mode='auto')
hist = my_model.fit_generator(steps_per_epoch=steps_per_epoch,generator=traindata, validation_data= testdata, validation_steps=validation_steps,epochs=30,callbacks=[checkpoint,early])

import matplotlib.pyplot as plt

plt.plot(hist.history["accuracy"])
plt.plot(hist.history['val_accuracy'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()


import pickle

class_dictionary =traindata.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}
print(class_dictionary)
# save the class dictionary to pickle
face_label_filename = 'models/face-labels.pickle'
with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)