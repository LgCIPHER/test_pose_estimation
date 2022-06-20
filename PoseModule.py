import cv2 as cv
import mediapipe as mp
import os.path as path
import time

class PoseDetector():
    def __init__(self, mode=False, complex=1, smooth=True, enableSeg=False,
                smoothSeg=True, detectCon=0.5, trackCon=0.5):
        
        self.mode = mode
        self.complex = complex
        self.smooth = smooth
        self.enableSeg = enableSeg
        self.smoothSeg = smoothSeg
        self.detectCon = detectCon
        self.trackCon = trackCon

        # Pose module from "mp"
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complex, self.smooth, self.enableSeg,
                                    self.smoothSeg, self.detectCon, self.trackCon)
    
    def findPose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:        
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                            self.mpPose.POSE_CONNECTIONS)
        
        return img

    def findPosition(self, img, draw=True):
        lmList = []
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 10, (255, 0, 255), cv.FILLED)

        return lmList

def main():
    # Frame rate
    pTime = 0
    cTime = 0

    # Path to videos folder
    dir_path = path.dirname(path.realpath(__file__))
    videos_folder_path = path.join(dir_path, "PoseVideos/")

    # Open up the video selected
    video_path = path.join(videos_folder_path, "1.mp4")
    cap = cv.VideoCapture(video_path)

    detector = PoseDetector()

    while True:
        success, img = cap.read()

        img = detector.findPose(img)
        lmList = detector.findPosition(img,draw=False)

        if len(lmList) != 0:
            print(lmList[0])
            cv.circle(img, (lmList[0][1], lmList[0][2]), 10, (255, 0, 255), cv.FILLED)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        fps_display = "FPS: " + str(int(fps))

        cv.putText(img, fps_display, (10, 50), cv.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv.imshow("Video", img)        
        if cv.waitKey(1) & 0xFF == ord('x'):
            break

if __name__ == "__main__":
    main()
