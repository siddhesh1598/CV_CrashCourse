# import
import numpy as np
import time
import argparse
import cv2

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Cadffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
    help="path to ImageNet labels")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

# load class labels
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# resize the image to 224x224 and
# mean subtract (104, 117, 123) to normalize the input
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# load the model
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# perform classification
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took: {:.5f} seconds".format(end - start))

# sort the prediction in descending order and grab top-5
idxs = np.argsort(preds[0])[::-1][:5]

# loop over the predictions and display them
for (i, idx) in enumerate(idxs):
    # draw the top prediction on the input image
    if i == 0:
        text = "Label: {}, {:.2f}%".format(classes[idx], preds[0][idx] * 100)
        cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2)

    # display the predicted label along with the prob in the console
    print("[INFO] {}. label: {}, prob: {:.2f}%".format(i+1, classes[idx],
            preds[0][idx] * 100))

# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
