import os
import subprocess

import WaymoDataToolkit

if __name__ == "__main__":
    
    # Variables to store the images and their labels
    imageDir = '../data/camera/images/*.png'
    labelDir = '../data/camera/labels/*.txt'
    imageCounter = 0

    # Variables that store the command to retrieve training, test and validation data    
    trainingData = 'gsutil ls gs://waymo_open_dataset_v_1_2_0_individual_files/training'

    # Changing the variable here will change the source of data being retrieved
    stream = os.popen(trainingData)
    output = stream.read()
    print('Fetched list of files from Waymo Bucket.')

    # Stored as a list of URLs
    urlList = output.split('\n')
    totalRecords = len(urlList)
    
    # Option to start from a different file, e.g. third file from the list or have a bound on the files with endFile
    startFile = 0
    endFile = totalRecords
    urlList = urlList[startFile:endFile]

    for url in urlList:
        # Clears the directory of any previous images or labels to avoid having any residual images from previous run
        delCmd = subprocess.call(["sudo", "rm", "-rf", imageDir, labelDir])
        dataset = WaymoDataToolkit.WaymoDataToolkit(url)
        dataset.dataRetriever()
        dataset.dataExtractor()
        # Remove the break to download all the files
        break