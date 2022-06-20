import cv2 as cv
import mediapipe as mp
import os.path as path
import time

# Frame rate
pTime = 0
cTime = 0

# Path to videos folder
dir_path = path.dirname(path.realpath(__file__))
videos_folder_path = path.join(dir_path, "PoseVideos/")

# Open up the video selected
video_path = path.join(videos_folder_path, "1.mp4")
cap = cv.VideoCapture(video_path)

# Pose module from "mp"
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
while True:
    success, img = cap.read()
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    # print(results.pose_landmarks)
    
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id == 0:
                print(id, cx, cy)
                cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)
        
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    fps_display = "FPS: " + str(int(fps))

    cv.putText(img, fps_display, (10, 50), cv.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv.imshow("Video", img)    
    if cv.waitKey(1) & 0xFF == ord('x'):
        break