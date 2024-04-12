from mylib import config, thread
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os, time, schedule
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
if config.USE_GPU:
# set CUDA as the preferable backend and target
print("")
print("[INFO] Looking for GPU")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
if not args.get("input", False):
print("[INFO] Starting the live stream..")
vs = cv2.VideoCapture(config.url)
if config.Thread:
cap = thread.ThreadingClass(config.url)
time.sleep(2.0)
else:
print("[INFO] Starting the video..")
vs = cv2.VideoCapture(args["input"])
if config.Thread:
cap = thread.ThreadingClass(args["input"])
writer = None
fps = FPS().start()
while True:
if config.Thread:
frame = cap.read()
else:
(grabbed, frame) = vs.read()
if not grabbed:
break
frame = imutils.resize(frame, width=700)
results = detect_people(frame, net, ln,
personIdx=LABELS.index("person"))
serious = set()
abnormal = set()
if len(results) >= 2:
centroids = np.array([r[2] for r in results])
D = dist.cdist(centroids, centroids, metric="euclidean")
for i in range(0, D.shape[0]):
for j in range(i + 1, D.shape[1]):
if D[i, j] < config.MIN_DISTANCE:
serious.add(i)
serious.add(j)
if (D[i, j] < config.MAX_DISTANCE) and notserious:
abnormal.add(i)
abnormal.add(j)
for (i, (prob, bbox, centroid)) in enumerate(results):
(startX, startY, endX, endY) = bbox
(cX, cY) = centroid
color = (0, 255, 0)
if i in serious:
color = (0, 0, 255)
elif i in abnormal:
color = (0, 255, 255) #orange = (0, 165, 255)
cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
cv2.circle(frame, (cX, cY), 5, color, 2)
Safe_Distance = "Safe distance: >{} px".format(config.MAX_DISTANCE)
cv2.putText(frame, Safe_Distance, (470, frame.shape[0] - 25),
cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
Threshold = "Threshold limit: {}".format(config.Threshold)
cv2.putText(frame, Threshold, (470, frame.shape[0] - 50),
cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 0, 0), 2)
text = "Total serious violations: {}".format(len(serious))
cv2.putText(frame, text, (10, frame.shape[0] - 55),
cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
text1 = "Total abnormal violations: {}".format(len(abnormal))
cv2.putText(frame, text1, (10, frame.shape[0] - 25),
cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
if len(serious) >= config.Threshold:
cv2.putText(frame, "-ALERT: Violations over limit-", (10, frame.shape[0] - 80),
cv2.FONT_HERSHEY_COMPLEX, 0.60, (0, 0, 255), 2)
if config.ALERT:
print("")
print('[INFO] Sending mail...')
Mailer().send(config.MAIL)
print('[INFO] Mail sent')
if args["display"] > 0:
# show the output frame
cv2.imshow("A Deep Learning Based Socail Distance monitoring framework for
COVID 19", frame)
key = cv2.waitKey(1) & 0xFF
if key == ord("q"):
break
fps.update()
if args["output"] != "" and writer is None:
# initialize our video writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter(args["output"], fourcc, 25,
(frame.shape[1], frame.shape[0]), True)
if writer is not None:
writer.write(frame)
fps.stop()
print("===========================")
print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
