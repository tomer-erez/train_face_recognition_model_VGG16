import cv2
import pickle
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/vgg16_1.h5")
labels = pickle.loads(open("models/face-labels.pickle", "rb").read())
print(labels)
vid = cv2.VideoCapture(0)
haar_cascade = cv2.CascadeClassifier('models\haarcascade_frontalface_default.xml')

while (True):

    # Capture the video frame
    # by frame
    ret, raw_image = vid.read()
    image=raw_image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_img=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.3, 5)
    if len(faces_rect)==0:
        print('no faces')
    elif len(faces_rect)>1:
        print('multiple faces')

    else:
        print('found one face')
        (x, y, w, h) = faces_rect[0]
        face_roi=image[y:y+h,x:x+w]
        face_roi = cv2.resize(face_roi, (224, 224))
        img = np.expand_dims(face_roi, axis=0)
        predictions = model.predict(x=img)  # change or remove type control maybe
        ind = np.argmax(predictions)
        label = labels[ind]
        for id, pred in enumerate(np.nditer(predictions)):
            print(labels[id], ': ', "{:.2f}".format(pred), '\t', end=" ")
        print('\nidentification: ', label, '\n\n\n')
        cv2.imshow('face', face_roi)
        cv2.waitKey(1)

        raw_image=cv2.putText(raw_image, label, (x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(255,128,0), thickness=2)

        raw_image = cv2.rectangle(raw_image, (x,y), (x+w,y+h), color=(0,128,255), thickness=2)

    cv2.imshow('frame', raw_image)

    cv2.waitKey(1)



vid.release()
cv2.destroyAllWindows()

