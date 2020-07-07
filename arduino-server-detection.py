from http.server import HTTPServer, BaseHTTPRequestHandler
import cgi, time, socket, socketserver, threading, cv2 
from pyfirmata import Arduino, util
from multiprocessing import Process, freeze_support
import numpy as np
from shapely.geometry import Polygon
#Global Parameters

speed = 0.5
forwardFactor = 1
reverseFactor = 0
auto = True

def detection():
    global speed
    net = cv2.dnn.readNet('cars_detection.weights', 'yolov3_testing.cfg')

    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()

    cap = cv2.VideoCapture('test4.mp4')
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    count = 0


    while True:
        _, img = cap.read()

        # to skip frames:
        count += 5
        if count > 610:
            count = 50
        print(count)
        cap.set(1,count)

        # resize the image to half
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        height, width, _ = img.shape

        # copy of original image
        overlay = img.copy()

        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        # adding polygons
        alpha = 0.4
        yellow_poly = np.array( [ [100,int(height/1.6)], [width-100,int(height/1.6)],[width,height-50], [width, height], [0,height], [0,height-50] ] )
        cv2.fillPoly(overlay, pts =[yellow_poly], color=(0,255,255))

        red_poly = np.array( [ [100,int(height/1.25)], [width-100,int(height/1.25)], [width-35, height], [35,height] ] )
        cv2.fillPoly(overlay, pts =[red_poly], color=(0,0,255))

        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        isNotRed = True
        alert = (0,255,0)

        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
                cv2.rectangle(img, (x,y), (x+w, y+h), (150,150,150), 1)
                cv2.putText(img, label , (x, y+20), font, 2, (0,0,0), 2)
                cv2.putText(img, label , (x-1, y+20), font, 2, (255,255,255), 2)

                focus_point = (int(x+w/2),y+h-3)
                yresult = cv2.pointPolygonTest(yellow_poly, focus_point, False)
                rresult = cv2.pointPolygonTest(red_poly, focus_point, False)

                rect = Polygon([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])           
                r_poly = Polygon([(100,int(height/1.25)),(width-100,int(height/1.25)),(width-35, height),(35,height)])
                y_poly = Polygon([(100,int(height/1.6)), (width-100,int(height/1.6)),(width,height-50), (width, height), (0,height), (0,height-50)])
                y_intersection = rect.intersects(y_poly)
                r_intersection = rect.intersects(r_poly)

                if y_intersection:
                    alert = (0,255,255)
                    if auto:
                        speed = 0.1
                if r_intersection and isNotRed:
                    isNotRed = False
                    alert = (0,0,255)
                    if auto:
                        speed = 0.003

        cv2.circle(img, (30,30), 15, alert, -1)
        cv2.circle(img, (30,30), 14, (255,255,255), 1)
        cv2.circle(img, (30,30), 15, (0,0,0), 1)  
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

def run_arduino():
    board = Arduino('COM1')
    iterator = util.Iterator(board)
    iterator.start()

    print("Serial Port Initialized...")
    fwd = board.get_pin('d:11:p')     
    rvs = board.get_pin('d:10:p')

    while True:
        fwd.write(speed*forwardFactor)
        rvs.write(speed*reverseFactor)

        if speed == 0.003:
            motion = False
        else:
            motion = True
        if forwardFactor:
            direction = "Forward"
        else:
            direction = "Reverse"
        print("Speed:", speed, "| Motion:", motion, "| Direction", direction)
        # if isForward and isMoving:
        #     rvs.write(0)
        #     fwd.write(speed)
        # elif isMoving:
        #     fwd.write(0)
        #     rvs.write(speed)
        # else:
        #     fwd.write(0)
        #     rvs.write(0)
        time.sleep(1)

def run_server():
    class requestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.path = "/index.html"
            try:
                file_to_open = open(self.path[1:]).read()
                self.send_response(200)
            except:
                file_to_open = "File Not Found"
                self.send_response(404)
            self.end_headers()
            self.wfile.write(bytes(file_to_open, 'utf-8'))

        def do_POST(self):
            global speed
            global forwardFactor
            global reverseFactor
            if self.path.endswith("/"):

                ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
                pdict['boundary'] = bytes(pdict['boundary'], 'utf-8')
                content_len = int(self.headers.get('Content-length'))
                pdict['CONTENT-LENGTH'] = content_len
                if ctype == 'multipart/form-data':
                    fields = cgi.parse_multipart(self.rfile, pdict)
                    btn = fields.get('bttn')[0]
                    print(btn)

                    #handle input
                    if btn == "Stop":
                        speed = 0.003
                    elif btn == "Go":
                        if speed == 0.003:
                            speed = 0.5
                    elif btn == "Fast":
                        speed = 0.7
                    elif btn == "Faster":
                        speed = 0.85
                    elif btn == "Fastest":
                        speed = 0.99
                    elif btn == "Slow":
                        speed = 0.55
                    elif btn == "Slower":
                        speed = 0.40
                    elif btn == "Slowest":
                        speed = 0.2
                    elif btn == "Forward":
                        forwardFactor = 1
                        reverseFactor = 0 
                    elif btn == "Reverse":
                        forwardFactor = 0
                        reverseFactor = 1

                self.path = "/index.html"
                try:
                    file_to_open = open(self.path[1:]).read()
                    self.send_response(200)
                except:
                    file_to_open = "File Not Found"
                    self.send_response(404)
                self.end_headers()
                self.wfile.write(bytes(file_to_open, 'utf-8'))

    ip = socket.gethostbyname(socket.gethostname())
    print("Server Hosted at", ip, ":8000")  
    addr = (ip, 8000)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    sock.bind(addr)
    sock.listen(5)

    class Thread(threading.Thread):
        def __init__(self,i):
            threading.Thread.__init__(self)
            self.i = i
            self.daemon = True
            self.start()
        
        def run(self):
            httpd = HTTPServer(addr, requestHandler,False)
            httpd.socket = sock
            httpd.server_bind = self.server_close = lambda self: None
            httpd.serve_forever()

    [Thread(i) for i in range(100)]
    hrs = 24
    time.sleep(3600*hrs)


server_thread = threading.Thread(target=run_server)
server_thread.start()
arduino_thread = threading.Thread(target=run_arduino)
arduino_thread.start()
detection_thread = threading.Thread(target=detection)
detection_thread.start()
