import cv2
import mediapipe as mp
import os
import argparse

def process_img(img, face_detection):
    H, W, _ = img.shape  # Define H and W within the function
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    
    print(out.detections)
    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            # Correct calculation for bounding box
            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            # Blur the faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

def process_image(filePath, face_detection):
    img = cv2.imread(filePath)
    if img is None:
        print(f"Error: Could not load image from {filePath}")
        return
    img = process_img(img, face_detection)
    
    output_path = os.path.join(output_dir, 'output.jpg')
    cv2.imwrite(output_path, img)
    print(f"Image saved to {output_path}")

def process_video(filePath, face_detection):
    cap = cv2.VideoCapture(filePath)
    if not cap.isOpened():
        print(f"Error: Could not open video file {filePath}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame of the video")
        return
    
    output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), 
                                   cv2.VideoWriter_fourcc(*'MP4V'), 
                                   25, 
                                   (frame.shape[1], frame.shape[0]))

    while ret:
        frame = process_img(frame, face_detection)
        output_video.write(frame)
        ret, frame = cap.read()
            
    cap.release()
    output_video.release()
    print(f"Video saved to {os.path.join(output_dir, 'output.mp4')}")

def process_webcam(face_detection):
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Error: Could not access the webcam")
        return

    ret, frame = cap.read()
    while ret:
        frame = process_img(frame, face_detection)
        cv2.imshow('Webcam Feed', frame)
        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()
        
    cap.release()
    cv2.destroyAllWindows()

output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

args = argparse.ArgumentParser()
args.add_argument("--mode", default='webcam')
args.add_argument("--filePath", default=None)

args = args.parse_args()

# Initialize face detection
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    if args.mode == "image" and args.filePath:
        process_image(args.filePath, face_detection)
    elif args.mode == "video" and args.filePath:
        process_video(args.filePath, face_detection)
    elif args.mode == "webcam":
        process_webcam(face_detection)
    else:
        print("Invalid mode or file path not provided for image/video mode.")
