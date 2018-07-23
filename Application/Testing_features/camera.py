import cv2 as cv

cap = cv.VideoCapture(0)
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
face_data = []

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(len(faces))
    # Display the resulting frame
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        crop_img = frame[y:y + h, x:x + w]
        crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
        crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
        face_data.append(crop_img)

    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()