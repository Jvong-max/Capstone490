import cv2
import dlib
import time
import mediapipe as mp
import sys

# Drawing outline of poses, tracking and detection is halved.
mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# detection, change to fullbody.xml
personCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Select what video
video = cv2.VideoCapture("videos/walk.mp4")

# create region of interest?

# ROI functions?


# Video size
WIDTH = 1280
HEIGHT = 720

# Function to estimate speed, have user just enter distance and time

def estimateSpeed(userInputDistance, userInputTime):
    getDistance = float(userInputDistance)
    getTime = float(userInputTime)
    speed = getDistance/getTime
    format(speed, '.3f')
    return speed


# Main function

def main():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    current = 0

    tracking = {}
    location1 = {}
    location2 = {}

    # User Input, add gui later?
    userInputDistance = input("Enter distance(Meters) traveled: ")
    userInputTime = input("Time Taken(seconds): ")

    speed = {}

    pTime = 0

    # Save video to capstone directory in .avi format and motion jpeg
    out = cv2.VideoWriter('result-video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (WIDTH, HEIGHT))

    # While camera or video is playing
    while True:
        rc, image = video.read()

        if type(image) == type(None):
            break

        # Change image/video and copy in order to draw landmarks
        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        imgRGB = cv2.cvtColor(resultImage, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        # Draw landmarks and save plot points into file
        if results.pose_landmarks:
            mpDraw.draw_landmarks(resultImage,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            orig_stdout = sys.stdout
            f = open('landmark-plots.txt','w')
            sys.stdout = f
            for id, lm in enumerate(results.pose_landmarks.landmark):
                print(id, lm)
            sys.stdout = orig_stdout
            f.close()

        # Attempt to plot
        # mpDraw.plot_landmarks(results.pose_world_landmarks,  mpPose.POSE_CONNECTIONS)

        frameCounter = frameCounter + 1
        # personIDtoDelete = []

        # Calculate Frames

        cTime = time.time()
        frames = 1/(cTime-pTime)
        pTime = cTime

        # Count for each person that enters frame
        for personID in tracking.keys():
            trackingQuality = tracking[personID].update(image)
            # ID will be assigned to person
            # if trackingQuality < 7:
                # personIDtoDelete.append(personID)

        # Deletes the tracker if not focusing on person anymore
        # for personID in personIDtoDelete:
        #     tracking.pop(personID, None)
        #     location1.pop(personID, None)
        #     location2.pop(personID, None)

        # Using fullbody cascade 
        if not (frameCounter % 10):
            # Convert image and use casacade to detect object
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            person = personCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
            # Assign the trackers
            for (_x, _y, _w, _h) in person:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                #x_bar = x + 0.5 * w
                #y_bar = y + 0.5 * h

                matchpersonID = None
                # Assign tracker to id
                for personID in tracking.keys():
                    trackedPosition = tracking[personID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    # t_x_bar = t_x + 0.5 * t_w
                    # t_y_bar = t_y + 0.5 * t_h

                # Tracks person
                if matchpersonID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))

                    tracking[current] = tracker
                    location1[current] = [x, y, w, h]

                    current = current + 1

        # Creates object rectangle around the four positions
        for personID in tracking.keys():
            trackedPosition = tracking[personID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)

            location2[personID] = [t_x, t_y, t_w, t_h]

        # Estiamting speed based on location of person
        for i in location1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = location1[i]
                [x2, y2, w2, h2] = location2[i]
                location1[i] = [x2, y2, w2, h2]
                # if positions aren't equal estimate speed
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    speed[i] = estimateSpeed(userInputDistance,userInputTime)
                    # Display speed
                    if speed[i] != None and y1 >= 1:
                        cv2.line(resultImage, (45,70), (255,70), (0,0,0), 28)
                        cv2.putText(resultImage, "Speed: " + str(format(speed[i], '.3f')) + " m/s", (50,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) ,2)
        # Display FPS
        cv2.putText(resultImage, "FPS: " + str(int(frames)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        # Waitkey might want to rework something else might be wrong
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    
    cv2.destroyAllWindows()
    out.release()

if __name__ == '__main__':
    main()