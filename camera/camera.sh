#!/bin/bash
cd /home/pi/tflite1
source tflite1-env/bin/activate
python3 TFLite_detection_webcam_lora_noscreen.py --modeldir=Sample_TFLite_model --edgetpu