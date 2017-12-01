from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import RMSprop, SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import numpy as np
from scipy.misc import imread,imresize

batch_size = 128
def load_image_directory():
  
  datagen = ImageDataGenerator(rescale=1. / 255)
  train_generator = datagen.flow_from_directory(
         '/Users/svdj16/Documents/FAU_project/train',
         target_size=(224, 224),
         batch_size=batch_size,
         class_mode='categorical',  # this means our generator will only yield batches of data, no labels
         shuffle=True,
         )

  validation_generator = datagen.flow_from_directory(
         '/Users/svdj16/Documents/FAU_project/validation',
         target_size=(224, 224),
         batch_size=batch_size,
         class_mode='categorical',  # this means our generator will only yield batches of data, no labels
         shuffle=True,
         )
  return train_generator,validation_generator
  

def load_VGG16_model():
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  print "Model loaded..!"
  print base_model.summary()
  return base_model

def extract_features_and_store(base_model,train_generator,validation_generator):
  
  x_generator = None
  y_lable = None

  batch = 0
  for x,y in train_generator:
     if batch == (80929/batch_size):
         break
     print "predict on batch:",batch
     batch+=1
     if x_generator==None:
        x_generator = base_model.predict_on_batch(x)
        y_lable = y

     else:
        x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
        y_lable = np.append(y_lable,y,axis=0)

  x_generator,y_lable = shuffle(x_generator,y_lable)
  np.save(open('face_x_VGG16.npy', 'w'),x_generator)
  np.save(open('face_y_VGG16.npy','w'),y_lable)

  batch = 0

  x_generator = None
  y_lable = None

  for x,y in validation_generator:
     if batch == (17198/batch_size):
         break
     print "predict on batch validate:",batch
     batch+=1
     if x_generator==None:
        x_generator = base_model.predict_on_batch(x)
        y_lable = y
     else:
        x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
        y_lable = np.append(y_lable,y,axis=0)

  x_generator,y_lable = shuffle(x_generator,y_lable)
  np.save(open('face_x_validate_VGG16.npy', 'w'),x_generator)
  np.save(open('face_y_validate_VGG16.npy','w'),y_lable)
  
  train_data = np.load(open('face_x_VGG16.npy'))
  train_labels = np.load(open('face_y_VGG16.npy'))
  train_data,train_labels = shuffle(train_data,train_labels)
  validation_data = np.load(open('face_x_validate_VGG16.npy'))
  validation_labels = np.load(open('face_y_validate_VGG16.npy'))
  validation_data,validation_labels = shuffle(validation_data,validation_labels)
  return train_data,train_labels,validation_data,validation_labels

def train_model(train_data,train_labels,validation_data,validation_labels):
  ''' used fully connected layers, SGD optimizer and 
      checkpoint to store the best weights'''

  model = Sequential()
  model.add(Flatten(input_shape=train_data.shape[1:]))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(43, activation='softmax'))
  sgd = SGD(lr=0.0005, decay = 1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  #model.load_weights('video_3_512_VGG_no_drop.h5')
  callbacks = [  ModelCheckpoint('face_vgg_checl.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
  nb_epoch = 500
  model.fit(train_data,train_labels,validation_data = (validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)

if __name__ == '__main__':
  train_generator,validation_generator = load_image_directory()
  base_model = load_VGG16_model()
  train_data,train_labels,validation_data,validation_labels = extract_features_and_store(base_model,train_generator,validation_generator)
  train_model(train_data,train_labels,validation_data,validation_labels)
