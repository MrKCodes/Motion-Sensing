import cv2,time

# creating first frame for comparison
first_frame=None

video = cv2.VideoCapture(0)
# 0 for first camera, 1 for second camera,
# or pass the video file
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:
#frame to capture video frame by frame
    check, frame = video.read()

    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Gaussian Blurred image
    gray=cv2.GaussianBlur(gray,(21,21),0)
    if first_frame is None:
        first_frame= gray
        continue
    # absolute difference between the first and current frame
    delta_frame=cv2.absdiff(first_frame,gray)

    #threshold of the delta Frame
    thresh_frame=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)


    faces=face_cascade.detectMultiScale(gray,
    scaleFactor=1.05, minNeighbors=5)
    #for detecting the faces in rectangle
    for x, y, w,  h in faces:
        frame=cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("Gray scale Frames",frame) # shows the first frame
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("threshold Frame",thresh_frame)

    key=cv2.waitKey(10)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
