import cv2,datetime

# creating first frame for comparison
first_frame=None
time=[]
status_list=[None,None]
status = 0
video = cv2.VideoCapture(0)
# 0 for first camera, 1 for second camera,
# or pass the video file
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

    (cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 5000:   #ignoring changes less than 5000px
            continue
        (x, y, w, h ) = cv2.boundingRect(contour)
        status = 1
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    # if status == 0:
    #     status_list.append(0)
    #     #print("Nothing here")
    # elif status == 1:
    #     status_list.append(1)
    #     #print("Something's moving")
    # if status_list[-1] is 1 and status_list[-2] is 0:
    #     time.append(datetime.datetime.now())



    cv2.imshow("Gray scale Frames",frame) # shows the first frame
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("threshold Frame",thresh_frame)

    key=cv2.waitKey(2)
    if key == ord('q'):
        break
video.release()
print(time)
cv2.destroyAllWindows()
