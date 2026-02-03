import mmap
import cv2
import numpy as np
import time

SHM_NAME = "Local\\YoloIPC_SharedMemory"
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CHANNELS = 3
IMAGE_SIZE = INPUT_WIDTH * INPUT_HEIGHT * CHANNELS
SHM_SIZE = IMAGE_SIZE + 1

print("Waiting for Producer...")
while True:
    try:
        shm = mmap.mmap(-1, SHM_SIZE, tagname=SHM_NAME, access=mmap.ACCESS_WRITE)
        break
    except:
        time.sleep(0.5)

print("Connected. Press q to quit.")

while True:
    while shm[0] != 1:
        time.sleep(0.001)

    shm.seek(1)
    image_data = shm.read(IMAGE_SIZE)

    frame = np.frombuffer(image_data, dtype=np.uint8).reshape((INPUT_HEIGHT, INPUT_WIDTH, CHANNELS))
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    shm.seek(0)
    shm.write(b'\x00')

    cv2.imshow("Output", frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

shm.close()
cv2.destroyAllWindows()
