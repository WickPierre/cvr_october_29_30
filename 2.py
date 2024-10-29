import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape
    print(height, width)
    frame[height // 2 - 400:height // 2 + 400, width // 2 - 400:width // 2 + 400, 2] = 1
    cv2.imshow('camera', frame)
    key = cv2.waitKey(1)
    if key == ord(" "):
        break

cv2.destroyAllWindows()
cap.release()


