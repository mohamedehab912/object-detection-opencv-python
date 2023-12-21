import cv2
import numpy as np
detecting = "bottle"
#sS
# Function to calculate distance between two points
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to draw line and calculate distances
def draw_distance_lines(img, box, object_name, reference_point=(0, 0)):
    # Draw bounding box
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    
    # Calculate organ point (e.g., top-left corner of the bounding box)
    organ_point = (box[0], box[1])
    
    # Draw line between organ point and reference point
    cv2.line(img, reference_point, organ_point, color=(255, 0, 0), thickness=2)
    
    # Calculate vertical and horizontal distances
    vertical_distance = calculate_distance((organ_point[0], reference_point[1]), organ_point)
    horizontal_distance = calculate_distance((reference_point[0], organ_point[1]), organ_point)
    print(horizontal_distance)

    # Display text with distances and object name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    cv2.putText(img, f"Object: {object_name}", (10, 30), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(img, f"Vertical: {vertical_distance:.2f}", (10, 60), font, font_scale, (0, 0, 255), font_thickness)
    cv2.putText(img, f"Horizontal: {horizontal_distance:.2f}", (10, 90), font, font_scale, (255, 0, 0), font_thickness)


# Function to process camera input
def Camera():
    classNames = []
    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)

    cam = cv2.VideoCapture(0)

    width = int(cam.get(3))
    height = int(cam.get(4))
    net.setInputSize(width, height)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
   
     
    while True:
        success, img = cam.read()
        if not success or img is None:
            print("Error: Failed to capture frame")
            break

        img_resized = cv2.resize(img, (width, height))
        classIds, confs, bbox = net.detect(img_resized, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if detecting == classNames[classId - 1]:
                    object_name = classNames[classId - 1]
                    draw_distance_lines(img, box, object_name) 
                    
                  
        fps=cam.get(cv2.CAP_PROP_FPS)
        #print(fps)
        cv2.imshow('let me die', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Call Camera() Function for video from the camera
Camera()