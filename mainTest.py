import cv2
from keras import models
from PIL import Image
import numpy as np

model = models.load_model('BrainTumor10EpochsCategorical.h5')


image_path = r'pred/pred11.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f'Image not found at: {image_path}')


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = Image.fromarray(image)
img = img.resize((64, 64))  
img = np.array(img)

input_img = np.expand_dims(img, axis=0)
input_img = input_img / 255.0  

predictions = model.predict(input_img)

if predictions.shape[1] == 2:
    result = np.argmax(predictions, axis=1)
else:
    result = (predictions > 0.5).astype(int)

print("Predicted Class:", result[0])
