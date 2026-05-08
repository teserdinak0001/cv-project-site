from ultralytics import YOLO
from matplotlib import pyplot as plt
import cv2

model = YOLO('traffic_objects.pt', verbose=False)
image_dir = r"C:\Python\UnsupervisedLearner\extracted\2026-04-19_06-56-50_00990.jpg"

results = model(image_dir, conf=0.50, verbose=False)
print(results[0].boxes)

img = results[0].plot()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 1)
ax.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.title("Traffic Object Detection (Figure 1)")
plt.show()