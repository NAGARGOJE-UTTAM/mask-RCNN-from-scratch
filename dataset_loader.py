import json
import cv2
import numpy as np

# =========================
# Load COCO Dataset
# =========================

ANNOTATION_FILE = "dataset/annotations_trainval2017/instances_val2017.json"
IMAGE_FOLDER = "dataset/val2017"

with open(ANNOTATION_FILE, "r") as file:
    coco_data = json.load(file)

# =========================
# Read Image Information
# =========================

images = coco_data["images"]

first_image = images[10]

image_id = first_image["id"]
file_name = first_image["file_name"]
width = first_image["width"]
height = first_image["height"]

print("Image ID :", image_id)
print("File Name:", file_name)
print("Width    :", width)
print("Height   :", height)

# =========================
# Load Image
# =========================

image_path = IMAGE_FOLDER + "/" + file_name

image = cv2.imread(image_path)

if image is None:
    print("Image not found!")
    exit()

print(type(image))
print(image.shape)

# =========================
# Read Categories
# =========================

categories = coco_data["categories"]

print("\nTotal Categories :", len(categories))
print("\nFirst Category")
print("----------------")
print(categories[0])

category_dict = {}

for category in categories:
    category_dict[category["id"]] = category["name"]

# =========================
# Read Annotations
# =========================

annotations = coco_data["annotations"]

print("\nTotal Annotations :", len(annotations))

print("\nFirst Annotation")
print("----------------")
print(annotations[0])

# =========================
# Find Objects in Current Image
# =========================

image_annotations = []

for annotation in annotations:

    if annotation["image_id"] == image_id:
        image_annotations.append(annotation)

print("\nObjects Found :", len(image_annotations))

# =========================
# Print Object Information
# =========================

for object_number, annotation in enumerate(image_annotations, start=1):

    category_name = category_dict[annotation["category_id"]]

    print("\nObject", object_number)
    print("Category            :", category_name)
    print("Bounding Box        :", annotation["bbox"])
    print("Area                :", annotation["area"])

    segmentation = annotation["segmentation"]

    if isinstance(segmentation, list):

        print("Segmentation Type   : Polygon")
        print("Number of Polygons  :", len(segmentation))

        total_points = 0

        for polygon in segmentation:
            total_points += len(polygon) // 2

        print("Coordinate Points   :", total_points)

    else:

        print("Segmentation Type   : RLE")

# =========================
# Draw Objects
# =========================

for annotation in image_annotations:

    category_name = category_dict[annotation["category_id"]]

    x, y, w, h = annotation["bbox"]

    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x2 = x + w
    y2 = y + h

    cv2.rectangle(
        image,
        (x, y),
        (x2, y2),
        (0, 255, 0),
        2
    )

    text_y = y - 10

    if text_y < 20:
        text_y = y + 20

    cv2.putText(
        image,
        category_name,
        (x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    segmentation = annotation["segmentation"]

    if isinstance(segmentation, list):

        # One mask per object (used later during training)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for polygon in segmentation:

            points = np.array(polygon, dtype=np.int32)
            points = points.reshape((-1, 2))

            # Draw polygon outline
            cv2.polylines(
                image,
                [points],
                True,
                (255, 0, 0),
                2
            )

            # Fill the object's mask
            cv2.fillPoly(
                mask,
                [points],
                255
            )

# =========================
# Display Result
# =========================

cv2.imshow("Mask R-CNN Dataset Viewer", image)

cv2.waitKey(0)
cv2.destroyAllWindows()