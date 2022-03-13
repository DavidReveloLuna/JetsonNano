import cv2
import jetson.inference
import jetson.utils
import numpy as np

net = jetson.inference.detectNet("ssd-mobilenet-v2",["--model=/my_project/ssd-mobilenet.onnx","--labels=/my_project/labels.txt","--input-blob=input_0","--output-cvg=scores","--output-bbox=boxes"])

camera = cv2.VideoCapture("/dev/video0")
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,480)

while True:
	r,frame = camera.read()
	img = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float))
	
	detections = net.Detect(img)
	n_obj = len(detections)
	for detect in detections:
		Id = detect.ClassID
		item = net.GetClassDesc(Id)
		
		if item == "pawn": color = (0,255,0)
		else: color = (0,0,255)

		cv2.putText(frame,item,(int(detect.Left),int(detect.Top)),cv2.FONT_HERSHEY_SIMPLEX,1,color,3)
		cv2.rectangle(frame,(int(detect.Left),int(detect.Top)),(int(detect.Right),int(detect.Bottom)),(255,0,0),1)

	cv2.imshow("window",frame)
	if cv2.waitKey(1) == ord("q"):
		break;


camera.release()
cv2.destroyAllWindows()
