import cv2
import pickle
import open3d as o3d

# Add labels and bounding boxes to verify the data
def process_image(image, labels):
    color = (0, 255, 0)
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    for label in labels:
        print(label)
        label_list = list(map(float, label.split(" ")))
        startPoint = (int(label_list[1]), int(label_list[2]))
        sizePoint = (int(label_list[1] + label_list[3]), int(label_list[2] + label_list[4]))
        image = cv2.rectangle(image, startPoint, sizePoint, color=(255, 0, 0), thickness=3)
    return image

if __name__ == "__main__":
    
    # visualize images
    saveDir = "../data" # directory where the data was extracted
    imageDir = "{}/camera/images".format(saveDir)
    labelDir = "{}/camera/labels".format(saveDir)
    frameNum = 0
    camera_list = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    
    for camera in camera_list:
        img = cv2.imread("{}/{}_{}.png".format(imageDir, frameNum, camera), cv2.IMREAD_UNCHANGED)
        label = open("{}/{}_{}.txt".format(labelDir, frameNum, camera), "r")
        cv2.imshow("{}_{}".format(frameNum, camera), img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # visualize range data

    laserDir = "{}/laser/images".format(saveDir)
    points = pickle.load(open("{}/{}_points.data".format(laserDir, frameNum), "rb"))
    cp_points = pickle.load(open("{}/{}_cp_points.data".format(laserDir, frameNum), "rb"))

    depth_3d_pcd = o3d.geometry.PointCloud()
    depth_3d_pcd.points = o3d.utility.Vector3dVector(points[0])
    o3d.visualization.draw_geometries([depth_3d_pcd])

