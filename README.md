# train_face_recognition_model_VGG16

training a face recognition model based on google's vgg16

download this repo

## install packages
```
pip install -r requirements.txt
```

important issue: go to site-packages/keras_vggface/models.py

change 
```
from keras.engine.topology import get_source_inputs
```
to
```
from keras.utils.layer_utils import get_source_inputs

```

## provide dataset

the directory hierarchy:

create a directory named images. inside it create a directory for each person you want to train on,

inside it put all the images of said person.
```
-images
|----input
||   ----person_a
          ----img1.jpg
          ----img2.jpg
          ----img320.png
          
||    ----person_b
          ----img1.jpg
          ----img2.jpg
          .
          .
          .
          ----img410.png
-models
-processing.py
-test.py
-train.py
```

## process images

```
python processing.py
```

this command will crop the face from each image and devide the cropped faces to train test and val folders. 


## train model
```
python train.py
```

## test model with webcam

```
python test.py
```




