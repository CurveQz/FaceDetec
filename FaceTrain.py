import numpy as np
import os
import cv2

def knn(X, y, z, k=1):
    d = np.sum((X - z) ** 2, axis=1)
    idx = np.argsort(d)[:k]
    cls, vote = np.unique(y[idx], return_counts=True)
    return cls[np.argmax(vote)]

y = []
X = []
for f in os.listdir():
    if not os.path.isfile(f) and not f.startswith("."):
        for i in os.listdir(f):
            if i.endswith(".jpg"):
                x = cv2.imread(f+"/"+i, 0)   # 0 = อ่านเป็นภาพขาวดำ
                X.append(x.flatten())
                y.append(f)
X = np.array(X, dtype=float)   # ใช้ float กันค่า uint8 ล้นตอนลบกัน
y = np.array(y)
print(X.shape)
print(np.unique(y, return_counts=True))

# ---------- ทายหน้าจากกล้องแบบ real-time ----------

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print('เปิดกล้อง... (กด Esc หรือ q เพื่อออก)')
fail = 0
while True:
    ret, frame = cap.read()
    if not ret:                # บางทีเฟรมแรกๆ ยังไม่มา ให้ลองใหม่
        fail += 1
        if fail > 100:
            print('อ่านภาพจากกล้องไม่ได้ -- ปิดโปรแกรมอื่นที่ใช้กล้องอยู่ (เช่น faceRec.py) แล้วรันใหม่')
            break
        continue
    fail = 0
    face = cv2.cvtColor(frame[100:380, 210:430, :], cv2.COLOR_BGR2GRAY)
    z = face.flatten().astype(float)
    name = str(knn(X, y, z, k=15))
    cv2.rectangle(frame, (210,100), (430,380), (0,255,0), 2)
    cv2.putText(frame, name, (210,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:   # q หรือ Esc = ออก
        break
    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:   # กด X ปิดหน้าต่าง = ออก
        break
cap.release()
cv2.destroyAllWindows()
