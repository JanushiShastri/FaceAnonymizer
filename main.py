import cv2
import mediapipe as mp

#read image
img_path = 'IMG_7386.jpg'
img = cv2.imread(img_path)

H, W, _ = img.shape
#detect faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
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

            # Draw the rectangle around the face
            # cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 10)
            
            # blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30,30))

        # cv2.imshow('img',img)
        # cv2.waitKey(0)




# save image

cv2.imwrite('output.jpg', img)