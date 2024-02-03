#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

from std_msgs.msg import String
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("src/object_detection_package/models/object_detection.h5")
object_detection_publisher = rospy.Publisher('object_detection_results', String, queue_size=10)
classes = ["Object Detected", "No Object"]

class ImageProcessor:
    def __init__(self):
        rospy.init_node("camera_subscriber", anonymous=True)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera_image", Image, self.image_callback)
        self.latest_image = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        # Store the latest image for processing
        self.latest_image = cv_image

    def start_processing(self):
        rate = rospy.Rate(1)  # Set your desired rate
        while not rospy.is_shutdown():
            if self.latest_image is not None:
                # Process the latest image
                self.process_image(self.latest_image)
                # Reset the latest_image to None after processing
                self.latest_image = None
            rate.sleep()

    def process_image(self, cv_image):
        # Your prediction and processing code here
        # This method will be called only when a new image is available
        image_size = (224, 224)
        image = cv2.resize(cv_image, image_size)

        predictions = model.predict(np.expand_dims(image, axis=0))
        pred = np.argmax(predictions)
        print(classes[pred])
        detection_result_msg = String()
        detection_result_msg.data = "Object Detected!"
        object_detection_publisher.publish(detection_result_msg)

        # Display the image (you can modify this part based on your needs)
        cv2.imshow("Image Subscriber", image)
        cv2.waitKey(1)
        pass

if __name__ == "__main__":
    try:
        processor = ImageProcessor()
        processor.start_processing()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

