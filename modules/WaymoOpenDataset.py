import os
import cv2
import glob
import pickle
import threading
import numpy as np
from urllib.parse import urlparse
import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from google.protobuf.json_format import MessageToDict

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

class ToolKit:
    
    def __init__(self, training_dir=None, testing_dir=None, validation_dir=None, save_dir=None):

        self.segment = None
        
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.validation_dir = validation_dir

        self.save_dir = save_dir

        self.camera_dir = self.save_dir + "/camera"
        self.camera_images_dir = self.camera_dir + "/images"
        self.camera_labels_dir = self.camera_dir + "/labels"
        if not os.path.exists(self.camera_dir):
            os.makedirs(self.camera_dir)
        if not os.path.exists(self.camera_images_dir):
            os.makedirs(self.camera_images_dir)
        if not os.path.exists(self.camera_labels_dir):
            os.makedirs(self.camera_labels_dir)

        self.laser_dir = self.save_dir + "/laser"
        self.laser_images_dir = self.laser_dir + "/images"
        self.laser_labels_dir = self.laser_dir + "/labels"
        if not os.path.exists(self.laser_dir):
            os.makedirs(self.laser_dir)
        if not os.path.exists(self.laser_images_dir):
            os.makedirs(self.laser_images_dir)
        if not os.path.exists(self.laser_labels_dir):
            os.makedirs(self.laser_labels_dir)

        self.camera_list = ["UNKNOWN", "FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]

    def assign_segment(self, segment):
        self.segment = segment
        self.dataset = tf.data.TFRecordDataset("{}/{}".format(self.training_dir, self.segment), compression_type='')

    def list_training_segments(self):
        seg_list = []
        for file in os.listdir(self.training_dir):
            if file.endswith(".tfrecord"):
                seg_list.append(file)
        return seg_list

    def list_testing_segments(self):
        pass

    def list_validation_segments(self):
        pass

    #########################################################################
    # Extract Camera Images and Labels
    #########################################################################
    
    # Extract Camera Image
    def extract_image(self, ndx, frame):
        for index, data in enumerate(frame.images):
            decodedImage = tf.io.decode_jpeg(data.image, channels=3, dct_method='INTEGER_ACCURATE')
            decodedImage = cv2.cvtColor(decodedImage.numpy(), cv2.COLOR_RGB2BGR)
            cv2.imwrite("{}/{}_{}.png".format(self.camera_images_dir, ndx, self.camera_list[data.name]), decodedImage)

    # Extract Camera Label
    def extract_labels(self, ndx, frame):
        for index, data in enumerate(frame.camera_labels):
            camera = MessageToDict(data)
            camera_name = camera["name"]
            label_file = open("{}/{}_{}.txt".format(self.camera_labels_dir, ndx, camera_name), "w")
            try:
                labels = camera["labels"]
                for label in labels:
                    x = label["box"]["centerX"]
                    y = label["box"]["centerY"]
                    width = label["box"]["width"]
                    length = label["box"]["length"]
                    x = x - 0.5 * length
                    y = y - 0.5 * width
                    obj_type = label["type"]
                    obj_id = label["id"]
                    label_file.write("{},{},{},{},{},{}\n".format(obj_type, x, y, length, width, obj_id))
            except:
                pass
            label_file.close()
    
    # Implemented Extraction as Threads
    def threaded_camera_image_extraction(self, datasetAsList, range_value):
        
        frame = open_dataset.Frame()
        
        for frameIdx in range_value:
            frame.ParseFromString(datasetAsList[frameIdx])
            self.extract_image(frameIdx, frame)
            self.extract_labels(frameIdx, frame)

    # Function to call to extract images
    def extract_camera_images(self):

        # clear images and labels from previous file
        self.delete_files(glob.glob("{}/*.png".format(self.camera_images_dir), recursive=True))
        self.delete_files(glob.glob("{}/*.txt".format(self.camera_labels_dir), recursive=True))
        open("{}/camera/last_file.txt".format(self.save_dir), 'w').write(self.segment)

        # Convert tfrecord to a list
        datasetAsList = list(self.dataset.as_numpy_iterator())
        totalFrames = len(datasetAsList)

        threads = []

        for i in self.batch(range(totalFrames), 30):
            t = threading.Thread(target=self.threaded_camera_image_extraction, args=[datasetAsList, i])
            t.start()
            threads.append(t)
        
        for thread in threads:
            thread.join()

    #########################################################################
    # Extract Laser Images and Labels
    #########################################################################

    # Extract LiDAR Image
    def extract_laser(self, ndx, frame):

        # Extract the range images, camera projects and top pose
        range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        frame.lasers.sort(key=lambda laser: laser.name)
        # Using the function provided by Waymo OD to convert range image into to point clouds to visualize and save the data
        points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)

        # Point cloud data
        f_p = open("{}/{}_points.data".format(self.laser_images_dir, ndx), 'wb')
        pickle.dump(points, f_p)
        # Camera projects for each point cloud data
        f_cp = open("{}/{}_cp_points.data".format(self.laser_images_dir, ndx), 'wb')
        pickle.dump(cp_points, f_cp)

    # Extract LiDAR Labels
    def extract_laser_labels(self, ndx, frame):
        f = open("{}/{}_labels.csv".format(self.laser_labels_dir, ndx), "w")
        for laser_label in frame.laser_labels:
            center_x = laser_label.box.center_x
            center_y = laser_label.box.center_y
            center_z = laser_label.box.center_z
            width = laser_label.box.width
            length = laser_label.box.length
            heading = laser_label.box.heading
            speed_x = laser_label.metadata.speed_x
            speed_y = laser_label.metadata.speed_y
            accel_x = laser_label.metadata.accel_x
            accel_y = laser_label.metadata.accel_y
            label_type = laser_label.type
            obj_id = laser_label.id
            num_points = laser_label.num_lidar_points_in_box
            # does not consider the level of difficulty to track the object, no requirement right now. maybe in the future
            f.write(
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(center_x, center_y, center_z, width, length, heading, speed_x,
                                                                speed_y, accel_x, accel_y, label_type, obj_id, num_points))
        f.close()        

    #  Implemented Extraction as Threads
    def threaded_laser_image_extraction(self, datasetAsList, range_value):
        
        frame = open_dataset.Frame()
        
        for frameIdx in range_value:
            frame.ParseFromString(datasetAsList[frameIdx])
            self.extract_laser(frameIdx, frame)
            self.extract_laser_labels(frameIdx, frame)

    # Function call to extract LiDAR images
    def extract_laser_images(self):

        # clear images and labels from previous file
        self.delete_files(glob.glob("{}/*.data".format(self.laser_images_dir), recursive=True))
        self.delete_files(glob.glob("{}/*.txt".format(self.laser_labels_dir), recursive=True))
        open("{}/laser/last_file.txt".format(self.save_dir), 'w').write(self.segment)

        # Convert tfrecord to a list
        datasetAsList = list(self.dataset.as_numpy_iterator())
        totalFrames = len(datasetAsList)

        threads = []

        for i in self.batch(range(totalFrames), 30):
            t = threading.Thread(target=self.threaded_laser_image_extraction, args=[datasetAsList, i])
            t.start()
            threads.append(t)
        
        for thread in threads:
            thread.join()

    #########################################################################
    # Save Video
    #########################################################################

    def process_image(self, image, labels):
        color = (0, 255, 0)
        for label in labels:
            label_list = list(map(str, label.split(",")))
            startPoint = (int(float(label_list[1])), int(float(label_list[2])))
            sizePoint = (int(float(label_list[1]) + float(label_list[3])), int(float(label_list[2]) + float(label_list[4])))
            image = cv2.rectangle(image, startPoint, sizePoint, color=(255, 0, 0), thickness=3)
            if label_list[0] == "TYPE_UNKNOWN":
                self.frame_type_unknown += 1
            elif label_list[0] == "TYPE_VEHICLE":
                self.frame_type_vehicle += 1
            elif label_list[0] == "TYPE_PEDESTRIAN":
                self.frame_type_ped += 1
            elif label_list[0] == "TYPE_SIGN":
                self.frame_type_sign += 1
            elif label_list[0] == "TYPE_CYCLIST":
                self.frame_type_cyclist += 1
        return image

    def write_text(self, image):
        font = cv2.FONT_HERSHEY_COMPLEX
        org1 = (50, 50)
        org2 = (50, 85)
        org3 = (50, 120)
        org4 = (50, 155)
        org5 = (50, 190)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 2

        image = cv2.putText(image, "Unknown: {}".format(self.frame_type_unknown), org1, font, fontScale, color, thickness,
                            cv2.LINE_AA)
        image = cv2.putText(image, "Vehicle: {}".format(self.frame_type_vehicle), org2, font, fontScale, color, thickness,
                            cv2.LINE_AA)
        image = cv2.putText(image, "Pedestrian: {}".format(self.frame_type_ped), org3, font, fontScale, color, thickness,
                            cv2.LINE_AA)
        image = cv2.putText(image, "Sign: {}".format(self.frame_type_sign), org4, font, fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, "Cyclist: {}".format(self.frame_type_cyclist), org5, font, fontScale, color, thickness,
                            cv2.LINE_AA)

        return image
    
    def save_video(self):

        if not os.path.isdir("{}/videos".format(self.save_dir)):
            os.makedirs("{}/videos".format(self.save_dir))

        cameraList = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
        totalFrames = len(glob.glob1(self.camera_images_dir, "*_FRONT.png"))
        self.frame_type_unknown = None
        self.frame_type_vehicle = None
        self.frame_type_ped = None
        self.frame_type_sign = None
        self.frame_type_cyclist = None
        # print("Found {} frames.".format(totalFrames))

        stat_data_file = open("{}/videos/{}.csv".format(self.save_dir, self.segment[:-9]), "w")

        img_array = []

        for i in range(totalFrames):

            self.frame_type_unknown = 0
            self.frame_type_vehicle = 0
            self.frame_type_ped = 0
            self.frame_type_sign = 0
            self.frame_type_cyclist = 0

            front_image_list = []
            for camera in cameraList[:3]:
                image = cv2.imread("{}/{}_{}.png".format(self.camera_images_dir, i, camera), cv2.IMREAD_UNCHANGED)
                label = open("{}/{}_{}.txt".format(self.camera_labels_dir, i, camera), "r")
                image = self.process_image(image, label)
                image = cv2.resize(image, (504, 336))
                front_image_list.append(image)
            front_view = np.hstack((front_image_list[0], front_image_list[1], front_image_list[2]))

            side_image_list = []
            for camera in cameraList[3:]:
                image = cv2.imread("{}/{}_{}.png".format(self.camera_images_dir, i, camera), cv2.IMREAD_UNCHANGED)
                label = open("{}/{}_{}.txt".format(self.camera_labels_dir, i, camera), "r")
                image = self.process_image(image, label)
                image = cv2.resize(image, (504, 231))
                side_image_list.append(image)
            data_image = np.zeros((231, 504, 3), np.uint8)
            data_image = self.write_text(data_image)
            side_view = np.hstack((side_image_list[0], data_image, side_image_list[1]))
            frame_view = np.vstack((front_view, side_view))
            height, width, layers = frame_view.shape
            img_array.append(frame_view)

            stat_data_file.write(
                "{},{},{},{},{},{}\n".format(i, self.frame_type_unknown, self.frame_type_vehicle, self.frame_type_ped, self.frame_type_sign,
                                            self.frame_type_cyclist))
            size = (width, height)
        out = cv2.VideoWriter("{}/videos/{}.avi".format(self.save_dir, self.segment[:-9]), cv2.VideoWriter_fourcc(*'DIVX'), 10,
                            size)
        stat_data_file.close()

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

    #########################################################################
    # Consolidate Object Count per Camera and frontal_velocity, weather, time and location
    #########################################################################
    def consolidate(self):

        if not os.path.isdir("{}/consolidation".format(self.save_dir)):
            os.makedirs("{}/consolidation".format(self.save_dir))

        # Convert tfrecord to a list
        datasetAsList = list(self.dataset.as_numpy_iterator())
        totalFrames = len(datasetAsList)

        frame = open_dataset.Frame()

        stat_file = open("{}/consolidation/{}.csv".format(self.save_dir, self.segment[:-9]), "w")
        
        for frameIdx in range(totalFrames):
            
            frame.ParseFromString(datasetAsList[frameIdx])

            front_list = []
            front_left_list = []
            front_right_list = []
            side_left_list = []
            side_right_list = []

            for index, data in enumerate(frame.camera_labels):
                type_unknown = 0
                type_vehicle = 0
                type_ped = 0
                type_sign = 0
                type_cyclist = 0
                camera = MessageToDict(data)
                camera_name = camera["name"]
                try:
                    labels = camera["labels"]
                except:
                    labels = None
                if labels is not None:
                    for label in labels:
                        if label["type"] == "TYPE_UNKNOWN":
                            type_unknown += 1
                        elif label["type"] == "TYPE_VEHICLE":
                            type_vehicle += 1
                        elif label["type"] == "TYPE_PEDESTRIAN":
                            type_ped += 1
                        elif label["type"] == "TYPE_SIGN":
                            type_sign += 1
                        elif label["type"] == 'TYPE_CYCLIST':
                            type_cyclist += 1
                    if camera_name == "FRONT":
                        front_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "FRONT_LEFT":
                        front_left_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "FRONT_RIGHT":
                        front_right_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "SIDE_LEFT":
                        side_left_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                    elif camera_name == "SIDE_RIGHT":
                        side_right_list = [type_unknown, type_vehicle, type_ped, type_sign, type_cyclist]
                else:
                    if camera_name == "FRONT":
                        front_list = [0, 0, 0, 0, 0]
                    elif camera_name == "FRONT_LEFT":
                        front_left_list = [0, 0, 0, 0, 0]
                    elif camera_name == "FRONT_RIGHT":
                        front_right_list = [0, 0, 0, 0, 0]
                    elif camera_name == "SIDE_LEFT":
                        side_left_list = [0, 0, 0, 0, 0]
                    elif camera_name == "SIDE_RIGHT":
                        side_right_list = [0, 0, 0, 0, 0]
            obj_list = front_list + front_left_list + front_right_list + side_left_list + side_right_list
            # determine the velocity
            velocity = MessageToDict(frame.images[0])
            stat_file.write("{},{},{},{},{}\n".format(','.join([str(obj_count) for obj_count in obj_list]), ','.join([str(vel) for vel in velocity["velocity"].values()]), frame.context.stats.weather, frame.context.stats.time_of_day, frame.context.stats.location))

    #########################################################################
    # Util Functions
    #########################################################################

    def delete_files(self, files):
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    def batch(self, iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]