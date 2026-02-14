import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
import glob
import os

from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


class CameraCalibration(Node):

    def __init__(self):
        super().__init__('camera_calibration')

        self.image_limit: int = 100
        self.image_counter: int = 0
        self.calibration_done: bool = False

        self.bridge = CvBridge()

        self.image_path = "images"
        self.full_image_path = os.path.join(
            get_package_share_directory("camera_calibration"),
            self.image_path
        )

        os.makedirs(self.full_image_path, exist_ok=True)

        self.sub_rgb_images = self.create_subscription(
            Image,
            "/camera/rgb_image",
            self._on_rgb,
            qos_profile_sensor_data
        )

        self.get_logger().info("Camera calibration node started.")


    def _on_rgb(self, msg: Image):

        if self.calibration_done:
            return

        if self.image_counter < self.image_limit:
            self.image_counter += 1
            self._save_image(msg)
        else:
            self.get_logger().info("Image limit reached. Starting calibration...")
            self.calibration_done = True

            self.destroy_subscription(self.sub_rgb_images)

            self.perform_calibration()

            self.get_logger().info("Calibration complete. Shutting down node.")
            rclpy.shutdown()


    def _save_image(self, image: Image):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except Exception as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        filename = f"image_{self.image_counter:04d}.jpg"
        save_path = os.path.join(self.full_image_path, filename)

        cv2.imwrite(save_path, cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        self.get_logger().info(
            f'Saved image {self.image_counter}/{self.image_limit}'
        )


    def perform_calibration(self):

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        chessboard_size = (9, 6)
        objp = np.zeros((9*6, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []

        images = glob.glob(os.path.join(self.full_image_path, "*.jpg"))

        if len(images) == 0:
            self.get_logger().error("No images found for calibration.")
            return

        gray = None

        for img_path in images:

            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(
                gray,
                chessboard_size,
                None
            )

            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )

                imgpoints.append(corners2)

        if len(objpoints) < 30:
            self.get_logger().error(
                f"Not enough valid chessboard detections for calibration, detected:{len(objpoints)}"
            )
            return

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            None,
            None
        )

        self.get_logger().info("===== CALIBRATION RESULTS =====")
        print("\nCamera Matrix:\n", mtx)
        print("\nDistortion Coefficients:\n", dist)
        print("\nReprojection Error:", ret)


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibration()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()