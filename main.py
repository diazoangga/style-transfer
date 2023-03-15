import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import argparse
from config_utils import Config
from IPython.display import display

import loss
import model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument("--content", help='Content File Name')
parser.add_argument("--style", help='Style File Name')
args = parser.parse_args()


CFG = Config(config_path='./parameters.yml')
MODEL_NAME = CFG.MODEL.MODEL_NAME
STYLE_PATH = CFG.INPUT.CONTENT_DIR
CONTENT_PATH = CFG.INPUT.STYLE_DIR
WEIGHT_PATH = CFG.MODEL.WEIGHT_PATH
RESULTS_PATH = CFG.INPUT.RESULTS_DIR
content_path = os.path.join(CONTENT_PATH, args.content)
style_path = os.path.join(STYLE_PATH,args.style)

ITERATION = CFG.HYPERPARAMETERS.ITERATION
# OPTIMIZER = CFG.HYPERPARAMETERS.OPTIMIZER
LEARNING_RATE = CFG.HYPERPARAMETERS.LEARNING_RATE
SAVE_EVERY_ITER = CFG.HYPERPARAMETERS.SAVE_EVERY_ITER

def preprocessing_img(path):
    img = cv2.imread(path)
    width, height = img.shape[0], img.shape[1]
    img_nrows = 640
    img_ncols = int(width*img_nrows/height)
    img = cv2.resize(img, [img_nrows,img_ncols])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.
    return img

# defining a function to clip the pixel values to the range 0-1 (used during training)
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor*255
    arr = np.array(tensor, dtype=np.uint8)
    arr = arr[0]
    return Image.fromarray(arr)
    
print("Read the content and style images from {} and {}...".format(content_path, style_path))
assert os.path.exists(style_path) == True, "{} doesn't exist".format(style_path)
assert os.path.exists(content_path) == True, "{} doesn't exist".format(content_path)
style_img = preprocessing_img(style_path)
content_img = preprocessing_img(content_path)

print("\nBuilding model...")
model_style = model.build_model(model_arch=MODEL_NAME, weights=WEIGHT_PATH)

# computing target content and style outputs
content_img_tensor = tf.expand_dims(tf.constant(content_img),axis=0)
_, target_content_output = loss.extract_style_content(content_img_tensor, model_style) # selecting content of content image 

style_img_tensor = tf.expand_dims(tf.constant(style_img),axis=0)
target_style_outputs,_ = loss.extract_style_content(style_img_tensor, model_style) # selecting style of style image

img = content_img.copy()
img = np.clip(img, 0. , 1.)
img = np.expand_dims(img, axis=0)

img = tf.Variable(img) # converting to tensorflow variable 
display(tensor_to_image(img))

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

print("Iteration begins...")
start = time.time()
str_time = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
results_folder = os.path.join(RESULTS_PATH, '{}_{}_{}'.format(args.content, args.style,str_time))
os.mkdir(results_folder)
for step in range(ITERATION):
    with tf.GradientTape() as tape :
        style_outputs, content_output = loss.extract_style_content(img, model_style)
        total_loss, content_loss, style_loss = loss.custom_loss(style_outputs, content_output, target_style_outputs, target_content_output)

        grad = tape.gradient(total_loss, img)
    optimizer.apply_gradients([(grad,img)])
    img.assign(clip_0_1(img))

    print("=",end='')
    if step % SAVE_EVERY_ITER == 0:
        tensor_to_image(img).save(os.path.join(results_folder, 'iter_{}.jpg'.format(step)))
        print(" Iter: {}".format(step),end='')
        print(" content loss:{}, style loss: {}, total loss: {}".format(content_loss, style_loss, total_loss))
end = time.time()

print("Total training time: {:.1f} seconds".format(end-start))
print("Results folder: {}".format(results_folder))