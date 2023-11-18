{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "83a47079",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1363a106",
   "metadata": {},
   "outputs": [],
   "source": [
    "face_cascade = cv2.CascadeClassifier('haarcascade_files/haarcascade_frontalface_default.xml')\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "while 1:\n",
    "    ret, img = cap.read()\n",
    "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    faces = face_cascade.detectMultiScale(gray, 1.3, 5)\n",
    "    for (x, y, w, h) in faces:\n",
    "        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)\n",
    "        \n",
    "    cv2.imshow('img', img)\n",
    "    if cv2.waitKey(1) & 0xFF == 27:\n",
    "        break\n",
    "        \n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e5c5506",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
