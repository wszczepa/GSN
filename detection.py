# import cv2
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator

# if __name__ == "__main__":
#     model = YOLO("D:\StudiaMagisterskie\GSN\license-plate-detection\model\\best.pt")
#     img = cv2.imread("D:\StudiaMagisterskie\GSN\license-plate-detection\original.jpg")
#     result = model.predict(img)[0]
#     annotator = Annotator(img)

#     for xyxy in result.boxes.xyxy:



#         roi = cv2.rectangle(img,(int(xyxy[0]), int(xyxy[1])),(int(xyxy[2]), int(xyxy[3])),(0,255,0))
#         # Extract the region of interest
#         #x, y, w, h = map(int, label[:4])
#         #roi = img[y:y+h, x:x+w]

#         # Save the extracted ROI with a unique filename
#         cv2.imwrite(f"output.jpg", roi)

# import cv2
# from ultralytics import YOLO
# from ultralytics.utils.plotting import Annotator

# if __name__ == "__main__":
#     model = YOLO("D:\StudiaMagisterskie\GSN\license-plate-detection\model\\best.pt")
#     img = cv2.imread("D:\StudiaMagisterskie\GSN\license-plate-detection\\Texas.jpg")
#     result = model.predict(img)[0]

#     for i, xyxy in enumerate(result.boxes.xyxy):
#         x1, y1, x2, y2 = map(int, xyxy[:4])

#         # Extract the region of interest
#         roi = img[y1-5:y2+5, x1-5:x2+5]

#         # Save the extracted ROI with a unique filename
#         cv2.imwrite(f"output_{i + 1}.jpg", roi)
        
import cv2
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("D:\StudiaMagisterskie\GSN\license-plate-detection\model\\best.pt")
    img = cv2.imread("D:\StudiaMagisterskie\GSN\license-plate-detection\\original.jpg")
    result = model.predict(img)[0]

    for i, xyxy in enumerate(result.boxes.xyxy):
        x1, y1, x2, y2 = map(int, xyxy[:4])

        roi = img[y1-5:y2+5, x1-5:x2+5]

        roi_resized = cv2.resize(roi, (300, 300))

        cv2.imwrite(f"output_{i + 1}.jpg", roi_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  