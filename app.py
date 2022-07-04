import cv2
import os
import time
import numpy as np
import requests
from flask import Flask, render_template, Response
from edge_impulse_linux.image import ImageImpulseRunner

app = Flask(__name__, static_folder='templates/assets')

runner = None
scaleFactor = 5
countPeople = 0
countPeopleList = [0]
inferenceSpeed = 0
videoCaptureDeviceId = int(1) # use 0 for web camera
use_soracom = False
  
def now():
    return round(time.time() * 1000)

def gen_frames():  # generate frame by frame from camera
    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, 'modelfile.eim')
    print('MODEL: ' + modelfile)
    global countPeople
    global countPeopleList
    global inferenceSpeed
    forwarding_inference_timer = 0
    uploading_image_timer = 0

    while True:
        
        with ImageImpulseRunner(modelfile) as runner:
            try:
                model_info = runner.init()
                print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
                labels = model_info['model_parameters']['labels']
                

                camera = cv2.VideoCapture(videoCaptureDeviceId)
                ret = camera.read()[0]
                if ret:
                    backendName = "dummy" #backendName = camera.getBackendName() this is fixed in opencv-python==4.5.2.52
                    w = camera.get(3)
                    h = camera.get(4)
                    print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                    camera.release()
                else:
                    raise Exception("Couldn't initialize selected camera.")
                
                next_frame = 0 # limit to ~10 fps here
                
                for res, img in runner.classifier(videoCaptureDeviceId):
                    count = 0
                    img = cv2.resize(img, (img.shape[1]*scaleFactor,img.shape[0]*scaleFactor))
                    
                    if (next_frame > now()):
                        time.sleep((next_frame - now()) / 1000)

                    # print('classification runner response', res)

                    if "classification" in res["result"].keys():
                        print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                        for label in labels:
                            score = res['result']['classification'][label]
                            print('%s: %.2f\t' % (label, score), end='')
                        print('', flush=True)

                    elif "bounding_boxes" in res["result"].keys():
                        # print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                        countPeople = 0
                        inferenceSpeed = res['timing']['classification']
                        for bb in res["result"]["bounding_boxes"]:
                            # print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                            if bb['value'] > 0:
                                countPeople = countPeople + 1
                                center_x = int(bb['x'] + bb['width']/2)*scaleFactor
                                center_y = int(bb['y'] + bb['height']/2)*scaleFactor
                                
                                img = cv2.circle(img, (center_x, center_y), 80, (255,0,0), -1)
                        
                    if(len(countPeopleList) < 12):
                        countPeopleList.insert(0,countPeople)
                    else:
                        countPeopleList.pop()
                        countPeopleList.insert(0,countPeople)
                    # print(countPeopleList)
                    
                    if(use_soracom):
                        if(((now() - forwarding_inference_timer)/1000) > 10):
                            send_inference()
                            forwarding_inference_timer = now()
                        
                        if(((now() - uploading_image_timer)/1000) > 60):
                            send_image(img)
                            uploading_image_timer = now()

                    ret, buffer = cv2.imencode('.jpg', img)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

                    next_frame = now() + 100
                    
            finally:
                if (runner):
                    runner.stop()

def get_ads():
    
    global countPeopleList
    ad0 = cv2.imread('templates/assets/0.png') 
    ad1 = cv2.imread('templates/assets/1.png')
    ad2 = cv2.imread('templates/assets/2.png')
    ad3 = cv2.imread('templates/assets/3.png')
    
    while True:  
        
        countPeople = round(np.average(countPeopleList))
        # print(countPeople)

        if countPeople == 0:
            ret, buffer = cv2.imencode('.jpg', ad0)

        elif countPeople == 1:
            ret, buffer = cv2.imencode('.jpg', ad1)
        
        elif countPeople == 2:
            ret, buffer = cv2.imencode('.jpg', ad2)
            
        else:
            ret, buffer = cv2.imencode('.jpg', ad3)
        
        
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # time.sleep(2)

def get_inference_speed():
    while True:
        # print(inferenceSpeed)
        yield "data:" + str(inferenceSpeed) + "\n\n"
        time.sleep(0.1)

def get_people():
    while True:
        # print(countPeople)
        yield "data:" + str(countPeople) + "\n\n"
        time.sleep(0.1)

def send_inference():
    global countPeople
    
    url = 'http://harvest.soracom.io'
    obj = {'countPeople' : countPeople}
    print(obj)
    x = requests.post(url, json = obj)

def send_image(image):
    
    cv2.imwrite('image.jpg', image)

    url = 'http://harvest-files.soracom.io'
    headers = {"content-type":"image/jpeg"}

    img = cv2.imread('image.jpg')
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.put(url, data=img_encoded.tostring(), headers=headers)
    # decode response
    #print(json.loads(response.text))


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ads_feed')
def ads_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(get_ads(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/inference_speed')
def inference_speed():
	return Response(get_inference_speed(), mimetype= 'text/event-stream')

@app.route('/people_counter')
def people_counter():
	return Response(get_people(), mimetype= 'text/event-stream')


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)