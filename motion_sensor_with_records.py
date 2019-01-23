import cv2, pandas
from datetime import datetime

# creating first frame for comparison
first_frame=None
time=[]
status_list=[None,None]
status = 0
video = cv2.VideoCapture(0)
df= pandas.DataFrame(columns=["START","END","DURATION"])
# 0 for first camera, 1 for second camera,
# or pass the video file
while True:
#frame to capture video frame by frame
    check, frame = video.read()
    status=0
    status_list.append(0)
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
        status_list.append(1)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        if status_list[-1]==1 and status_list[-2] == 0:
            time.append(datetime.now())
        elif status_list[-1] ==0 and status_list[-2] ==1:
            time.append(datetime.now())


    cv2.imshow("Gray scale Frames",frame) # shows the first frame
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("threshold Frame",thresh_frame)
    key=cv2.waitKey(2)
    if key == ord('q'):
        if status==1:   # if the object enters but not leaves
            time.append(datetime.now())
        break

#print(time)
for i in range(0,len(time),2):
    df=df.append({"START":time[i],"END":time[i+1],"DURATION":time[i+1]-time[i]},ignore_index=True)
df.to_csv("Records.csv")
#print(df)
video.release()
cv2.destroyAllWindows()
