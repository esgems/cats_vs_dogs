import cv2

class Viz():

	def show1(self, image, box1):
		cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), 3)
		cv2.imshow("image", image)
		cv2.waitKey()
		
	def show2(self, image, box1, box2):
		cv2.rectangle(image, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), 3)
		cv2.rectangle(image, (box2[0], box2[1]), (box2[2], box2[3]), (0, 255, 0), 3)
		cv2.imshow("image", image)
		cv2.waitKey()