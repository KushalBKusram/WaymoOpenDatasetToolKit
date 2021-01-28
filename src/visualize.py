import cv2
import pickle
import open3d as o3d

if __name__ == "__main__":
    
    # visualize images
    imageDir = "../data/camera/images"
    frameNum = 0
    camera_list = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
    """
    for camera in camera_list:
        img = cv2.imread("{}/{}_{}.png".format(imageDir, frameNum, camera), cv2.IMREAD_UNCHANGED)
        cv2.imshow("{}_{}".format(frameNum, camera), img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    """

    # visualize range data

    rangeDir = "../data/range"
    points = pickle.load(open("{}/{}_points.data".format(rangeDir, frameNum), "rb"))
    cp_points = pickle.load(open("{}/{}_cp_points.data".format(rangeDir, frameNum), "rb"))

    depth_3d_pcd = o3d.geometry.PointCloud()
    depth_3d_pcd.points = o3d.utility.Vector3dVector(points[0])
    o3d.visualization.draw_geometries([depth_3d_pcd])

