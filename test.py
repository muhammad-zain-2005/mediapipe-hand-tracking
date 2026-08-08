import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw=mp.solutions.drawing_utils

while True:
    success, frame = cap.read()

    if not success:
        break

    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    print(result.multi_hand_landmarks)
    if result.multi_hand_landmarks:
        for handlandm in result.multi_hand_landmarks:
            for id,lm in enumerate(handlandm.landmark):
                
                h,w,c=frame.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if id==4:
                    cv2.circle(frame,(cx,cy),25,(255,0,255),cv2.FILLED)
            mpdraw.draw_landmarks(frame,handlandm,mphands.HAND_CONNECTIONS)
            

    cv2.imshow("webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()