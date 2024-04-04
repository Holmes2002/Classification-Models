import gradio as gr
import onnxruntime as ort
import numpy as np
import cv2

def inference(image):
    labels = ["Birth",
    "CIC",
    "Diploma",
    "Transcript",
    "TTCT"]
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))/255.
    image = (image-IMAGENET_DEFAULT_MEAN)/IMAGENET_DEFAULT_STD
    image = image.transpose((2,0,1))
    image = np.array([image],dtype="float32")
    ort_sess = ort.InferenceSession('deit.onnx')
    input_name = ort_sess.get_inputs()[0].name
    output_name = ort_sess.get_outputs()[0].name

    outputs = ort_sess.run(None, {input_name: image})
    outputs = np.exp(outputs[0][0])
    outputs = outputs/ np.sum(outputs)
    predicted = labels[outputs.argmax(0)]
    return predicted
path = "2.jpg"
img = cv2.imread(path)
inference(img)