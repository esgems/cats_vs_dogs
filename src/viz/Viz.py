import cv2

class Viz():

	@staticmethod
	def show1(image, box1, name):
		cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), 3)
		cv2.putText(image, name, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
		cv2.imshow("image", image)
		cv2.waitKey()