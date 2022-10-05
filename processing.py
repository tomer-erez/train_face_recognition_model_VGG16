import cv2
import os
from os import path
import shutil
import pickle
import numpy as np
import time

with open('classes.txt', 'w') as f:
    for person in os.listdir('images/input'):
        f.write(person+'\n')



def process_images():
    """
    this function iterates through all of our classes folders, crops the face in each image
    and saves each cropped face instead of the original image
    :return:
    """
    headshots_folder_name = 'images/input'

    # dimension of images
    image_width = 224
    image_height = 224

    # for detecting faces
    facecascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    # set the directory containing the images
    images_dir = os.path.join(".", headshots_folder_name)
    print(images_dir)
    current_id = 0
    label_ids = {}
    start=time.time()
    # iterates through all the files in each subdirectories


    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg") \
                    or file.endswith("JPG") or file.endswith("JEPG"):
                print(root,file)
                path = os.path.join(root, file)
                print("file", file)

            # get the label name (name of the person)
            label = os.path.basename(root).replace(" ", ".").lower()

            # add the label (key) and its number (value)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            # load the image
            imgtest = cv2.imread(path)
            image_array = np.array(imgtest, "uint8")

            # get the faces detected in the image
            faces = facecascade.detectMultiScale(imgtest,
                scaleFactor=1.3, minNeighbors=5)

            # if not exactly 1 face is detected, skip this photo
            if len(faces) != 1:
                print('XXXXXX\n')
            # remove the original image
                os.remove(path)
                continue
            print("face detected\n\n")
            # save the detected face(s) and associate
            # them with the label
            for (x_, y_, w, h) in faces:

                # draw the face detected
                face_detect = cv2.rectangle(imgtest,
                        (x_, y_),
                        (x_+w, y_+h),
                        (255, 0, 255), 2)
                # plt.imshow(face_detect)
                # plt.show()

                # resize the detected face to 224x224
                size = (image_width, image_height)

                # detected face region
                roi = image_array[y_: y_ + h, x_: x_ + w]

                # resize the detected head to target size
                resized_image = cv2.resize(roi, size)
                colored_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                image_array = np.array(colored_image, "uint8")

                # remove the original image
                os.remove(path)

                # replace the image with only the face
                im = Image.fromarray(image_array)
                im.save(path)

    stop=time.time()
    print("time: ", stop-start," seconds")
process_images()


def train_test_split():
    """
    splits the data to train, test, val based on the file number
    """
    names=[]
    with open('classes.txt') as f:
        while True:
            line = f.readline()
            if not line:
                break
            name=line.strip()
            names.append(name)
    print(names)

    directory = ("train_cropped","test_cropped","val_cropped")
    images_dir = "images"

    for dir in directory:
        path = os.path.join(images_dir, dir)
        os.mkdir(path)
        print(path)


    for name in names:

        test_dir = "images/test_cropped"
        test_path = os.path.join(test_dir, name)
        os.mkdir(test_path)

        train_dir = "images/train_cropped"
        train_path = os.path.join(train_dir, name)
        os.mkdir(train_path)

        val_dir = "images/val_cropped"
        val_path = os.path.join(val_dir, name)
        os.mkdir(val_path)

        src = "images/input/"+name
        test = "images/test_cropped/"+name
        train = "images/train_cropped/"+name
        val= "images/val_cropped/"+name
        images = os.listdir(src)

        for index,image in enumerate(images):

            if index%4==0:
                new_path = shutil.move(f"{src}/{image}", test)
            elif index%15==0:
                new_path = shutil.move(f"{src}/{image}", val)
            else:
                new_path = shutil.move(f"{src}/{image}", train)

    print('done')

train_test_split()

