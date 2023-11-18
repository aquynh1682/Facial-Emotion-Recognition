from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
path = 'haarcascade_files/haarcascade_frontalface_default.xml'
# new_model = load_model('models/model_ensmodel.h5')
# new_model = load_model('models/models1.h5')
# new_model = load_model('models/models2.h5')
# new_model = load_model('models/models3.h5')
new_model = load_model('models/models5.h5')
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

EMOTIONS = ["angry", "disgust", "fear","happy", "sad", "surprised", "neutral"]

def get_label(argument):
    labels = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad' , 5:'Surprise', 6:'Neutral'}
    return(labels.get(argument, "Invalid emotion"))

rectangle_bgr = (255, 255, 255)

img = np.zeros((500, 500))
text = "Some text in a box!"

(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness = 1)[0]

text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color = (0,0,0), thickness=1)

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width = 800, height = 1300)
    faceCascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray = gray[y:y + h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0,0), 2)
        face = faceCascade.detectMultiScale(roi_gray)
        if len(face) == 0:
            print("face not detect")
        else:
            for (ex, ey, ew, eh) in face:
                face_roi = roi_color[ey: ey+eh, ex:ex + ew]

    final_image = cv2.resize(face_roi, (48, 48))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0

    font = cv2.FONT_HERSHEY_SIMPLEX

    predictions = new_model.predict(final_image)

    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN
    result_num = np.argmax(predictions)
    status = get_label(result_num)
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, predictions)):
        angry = "{}: {:.2f}%".format("angry", prob[0] * 100)
        disgust = "{}: {:.2f}%".format("fear", prob[1] * 100)
        scared = "{}: {:.2f}%".format("scared", prob[2] * 100)
        happy = "{}: {:.2f}%".format("happy", prob[3] * 100)
        sad = "{}: {:.2f}%".format("sad", prob[4] * 100)
        surprised = "{}: {:.2f}%".format("surprised", prob[5] * 100)
        neutral = "{}: {:.2f}%".format("neutral", prob[6] * 100)
        x1, y1, w1, h1 = 0,0,150,75
#         cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 +h1), (0, 0, 0), -1)
        cv2.putText(frame,angry , (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        x2, y2, w2, h2 = 0,30,150,75
#         cv2.rectangle(frame, (x2, x2), (x2 + w2, y2 +h2), (0, 0, 0), -1)
        cv2.putText(frame,disgust , (x2 + int(w2/10), y2 + int(h2/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0,2550), 2)

        x3, y3, w3, h3 = 0,60,150,75
        cv2.putText(frame,scared , (x3 + int(w3/10), y3 + int(h3/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        x4, y4, w4, h4 = 0,90,150,75
        cv2.putText(frame,happy , (x4 + int(w4/10), y4 + int(h4/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        x5, y5, w5, h5 = 0,120,150,75
        cv2.putText(frame,sad , (x5 + int(w5/10), y5 + int(h5/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        x6, y6, w6, h6 = 0,150,150,75
        cv2.putText(frame,surprised , (x6 + int(w6/10), y6 + int(h6/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        x7, y7, w7, h7 = 0,180,150,75
        cv2.putText(frame,neutral , (x7 + int(w7/10), y7 + int(h7/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, status, (x, y - 10), font, 2,(0,255,0), 2, cv2.LINE_4)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0))

    cv2.imshow('face Emotion Recognition', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()