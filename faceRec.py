import cv2
import os

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
name = 'Nut'
os.makedirs(name, exist_ok=True)
i = len(os.listdir(name)) + 1   # ต่อเลขจากภาพที่มีอยู่ ไม่ทับของเดิม
while True:
    ret, frame = cap.read()
    cv2.rectangle(frame, (210,100), (430,380), (0,0,255), 2)
    face = cv2.cvtColor(frame[100:380, 210:430, :], cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', frame)
    cv2.imshow('face', face)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'{name}/{i}.jpg', face)
        i += 1
    elif key == ord('q') or key == 27:   # q หรือ Esc = ออก
        break
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:   # กด X ปิดหน้าต่าง = ออก
        break
cap.release()
cv2.destroyAllWindows()
