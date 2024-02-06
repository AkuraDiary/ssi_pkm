import cv2


def initFaceRecognition():
    faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # set width
    cap.set(4, 480)  # set height
    runFaceRecognition(cap, faceCascade)


def runFaceRecognition(capture, faceCascade):
    while (True):
        ret, img = capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
        cv2.imshow('video', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # press ESC to quit
            finish(capture)
            break


def finish(capture):
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Press ESC to quit")
    initFaceRecognition()
