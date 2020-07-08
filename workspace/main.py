"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network
#from imutils.video import FPS

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # set threshhold for detection
    prob_threshold = args.prob_threshold
    
    model=args.model
    DEVICE=args.device
    CPU_EXTENSION=args.cpu_extension
    

    ### TODO: Load the model through `infer_network` ###

    infer_network.load_model(model, CPU_EXTENSION, DEVICE)
    network_shape=infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    #check for webcam
    if args.input=='CAM':
        input_plugin=0
    #check for image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode=True
        input_plugin=args.input
    else:
        input_plugin=args.input
        assert os.path.isfile(args.input), "file does not exist"
    
    

    cap=cv2.VideoCapture(input_plugin)
    cap.open(input_plugin)
    
    
    w=int(cap.get(3))
    h=int(cap.get(4))
    
    input_shape=network_shape['image_tensor']
    
    
    #variables
   
    total_count=0
    previous_count=0
    current_count=0
    duration=0
    time_unseen=0
    time_entered=0
    request_id=0
    frame_count=0
    ### TODO: Loop until stream is over ###
    fps = cap.get(cv2.CAP_PROP_FPS)
   
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        p_frame=cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame=p_frame.transpose((2, 0, 1))
        p_frame=p_frame.reshape(1, *p_frame.shape)
        

        ### TODO: Start asynchronous inference for specified request ###
        net_input={'image_tensor': p_frame, 'image_info': p_frame.shape[1:]}
        
        inf_start=time.time()
        infer_network.exec_net(net_input, request_id)
        
        frame_count +=1

        ### TODO: Wait for the result ###
        if infer_network.wait()==0:
            
            inf_end=time.time()
            inf_time=inf_end-inf_start
        

            ### TODO: Get the results of the inference request ###
            net_output=infer_network.get_output()
            num_boxes=0

            ### TODO: Extract any desired stats from the results ###
            inf_time_message = "Inference time: {:.3f}ms".format(inf_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            probabs=net_output[0,0,:,2]
            for i, p in enumerate(probabs):
                if p>prob_threshold:
                    
                    box=net_output[0, 0, i, 3:]
                    p1=(int(box[0]*w), int(box[1]*h))
                    p2=(int(box[2]*w), int(box[3]*h))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
                    
                    num_boxes +=1
                    
            #get statistics
            #person has entered frame
            if num_boxes > current_count:
                start_time=frame_count/fps
                time_entered +=1
                #if time_entered>2:
                previous_count=current_count
                current_count=num_boxes
                #client.publish("person", json.dumps({"count":current_count}), qos=0, retain=False)
                #time_entered=0
            #person has left frame       
            if num_boxes < current_count:
                if time_unseen>3:
                    previous_count=current_count
                    current_count=num_boxes
                    total_count +=previous_count - current_count
                    
                    end_time=frame_count/fps
                    duration=end_time-start_time
                    time_entered=0
                    #client.publish("person", json.dumps({"total": total_count}), qos=0, retain=False)
                    client.publish("person/duration", json.dumps({"duration":duration}), qos=0, retain=False)
                else:
                    time_unseen +=1
                
            client.publish("person", json.dumps({"count":current_count}), qos=0, retain=False)
                    
           
            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###

        ### TODO: Send the frame to the FFMPEG server ###
        frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
    cap.release()
    cv2.destroyAllWindows()
  
    
    
    


def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()