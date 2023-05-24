#Dung cho Qt5
import sys
from time import time
import cv2
import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap
# pip install pyqt5, pip install pyqt5 tools
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5 import QtGui
# just change the name
from Screen import Ui_MainWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # the way app working
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        #Start
        self.uic.pushButton_Stop.clicked.connect(self.take_pics)
        # self.uic.pushButton_Stop.clicked.connect(self.stop_capture_video)
        self.uic.pushButton_Browser.clicked.connect(self.linkto)
        self.str = "a"
        self.count = 0
        self.thread = {} #tao thu vien rong

    def linkto(self):
        #tim duong dan
        if self.count%2 == 1:
            self.stop_capture_video()
        self.count+=1
        link = QFileDialog.getOpenFileName(filter='*.mp4')
        #bat man hinh
        self.uic.lineEdit.setText(link[0])
        # Chay ham
        self.str = link[0]
        self.uic.pushButton_Start.clicked.connect(self.start_capture_video)


    def take_pics(self):
        self.thread[1].take_pic()

    def closeEvent(self, event):
        return self.stop_capture_video()
    
    def stop_capture_video(self):
        self.thread[1].stop()#tat thread1
        # self.thread[1].quit()
        # self.thread[1].wait()
        

    def start_capture_video(self):# bat dau video ghi hinh, nhan dang
        self.thread[1] = live_stream(index=1,str=self.str)
        self.thread[1].start() # khoi chay class
        self.thread[1].signal.connect(self.show_webcam)#tin hieu tra ve tu class co thi chay lenh
        self.thread[1].signal_1.connect(self.show_pic)

    def show_pic(self,pic):
        qt_img = self.convert_cv_qt(pic)
        self.uic.label.setPixmap(qt_img)

    def show_webcam(self,cv_img):#lay hinh show len PyQt
        qt_img = self.convert_cv_qt(cv_img)
        self.uic.label_Screen.setPixmap(qt_img)

    def convert_cv_qt(self,cv_img):
        rgb_image = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        h,w,ch = rgb_image.shape
        bytes_per_line = ch*w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data,w,h,bytes_per_line,QtGui.QImage.Format_RGB888)#doi tin hieu anh tu cv2 sang pyQt
        p = convert_to_Qt_format.scaled(700,500,Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)#tra ve tin hieu hinh anh da chuye ndoi xong


class live_stream(QThread):#khai bao Qthread moi truyen di tin hieu duoc (khoi chayj luong)
    signal = pyqtSignal(np.ndarray)#nd array la ma tran chua anh, khai bao du lieu tra ve dang anh
    signal_1 = pyqtSignal(object)
    def __init__(self,index,str,parent=None):
        self.pic = False
        self.inedx = index
        self.str_live = str
        self.count = 0
        print("Start threading",self.inedx)
        # print(self.str_live)
        super(live_stream,self).__init__(parent)# khang dinh lop live_stream thua huong het Qthread
        self.status = True
        
    def run(self):
        while self.status:
            self.model = self.load_model()
            self.classes = self.model.names
            self.out_file = "Labeled_Video.avi"
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu' #khong co cad thi chay cpu
            self.run_program()

    def load_model(self):#load ra model sau khi train tren colab
        # model = torch.hub.load('ultralytics/yolov5','yolov5s',pretrained=True)#Load model YOLOv5
        model = torch.hub.load('ultralytics/yolov5','custom',path='C:/Users/khina/OneDrive/Desktop/TestQt/YOLO_Qt/Big_exercise/best_8_tieng.pt')#load model cusstom
        return model
    
    def get_video_from_url(self):   
        # return cv2.VideoCapture("C:/Users/khina/OneDrive/Desktop/TestQt/YOLO_Qt/A21_YOLO_Detectet_video_on/Animals_cut.mp4")
        return cv2.VideoCapture(self.str_live)
    def score_frame(self,frame):#predict
        self.model.to(self.device)#lua CPU or GPU 
        frame = [frame]
        results = self.model(frame)#ket qua du doan
        labels, cord = results.xyxyn[0][:,-1].numpy(),results.xyxyn[0][:,:-1].numpy()
        return labels, cord # labels = class (so 0 , 1 ,2 ,3, ...), cord = toa do boundingbox
    
    def class_to_label(self,x):#tra ve ten tu vi tri index
        return self.classes[int(x)]
    
    def plot_boxes(self,results,frame):
        labels,cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):#co bao nhieu vat ve bang day khung
            row = cord[i]#lay tung khung 1 # tra ve toa do bongdingbox tinh bang % của frame
            if self.class_to_label(labels[i]) == "Horse" :
                if row[4] >= 0.6:
                    x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                    bgr = (0,255,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                    cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
            elif labels[i] == "Mule" :#do tin cay > 0.2 thi ve:
                if row[4] >= 0.7:
                    x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                    bgr = (0,255,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                    cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
            elif labels[i] == "Camel" :#do tin cay > 0.2 thi ve:
                if row[4] >= 0.2:
                    x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                    bgr = (0,255,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                    cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
            elif labels[i] == "Butterfly" :#do tin cay > 0.2 thi ve:
                if row[4] >= 0.7:
                    x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                    bgr = (0,255,0)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                    cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
            elif row[4] >= 0.4:#do tin cay > 0.2 thi ve:
                x1,y1,x2,y2 = int(row[0]*x_shape),int(row[1]*y_shape),int(row[2]*x_shape),int(row[3]*y_shape)
                bgr = (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),bgr,2)
                cv2.putText(frame,self.class_to_label(labels[i]),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),2)
        return frame

    def run_program(self):
        player = self.get_video_from_url()#lay frame video
        assert player.isOpened()#Lay tung fame cua video
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))#Lay chieu rong khung hinh
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))#lay chieu cao khung hinh
        four_cc = cv2.VideoWriter_fourcc(*"MJPG")#code ghi lai video
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape,y_shape))#ghi lai phim
        while True:
            start_time = time()#lay thoi diem bat dau
            ret,frame = player.read()#doc hinh anh
            assert ret # giong if else
            result = self.score_frame(frame)#du doan hinh anh luob
            frame = self.plot_boxes(result,frame)
            end_time = time()
            fps = 1/(np.round(end_time-start_time,3))#so hinh doan duoc tren 1 s
            print(f"Frames Per Second:{round(fps,2)} FPS ")
            #tra tin anh sau detetion ve 
            self.signal.emit(frame)
            if self.pic:
                self.signal_1.emit(frame)
                cv2.imwrite("C:/Users/khina/OneDrive/Desktop/TestQt/YOLO_Qt/Big_exercise/Save_pic/{}.jpg".format(self.count),frame)
                print(self.count)
                self.count+=1
                self.pic = False

    def take_pic(self):
        self.pic = True

    def stop(self):
        print("stop threading",self.inedx)
        self.status=False
        # self.wait()
        self.terminate()
        

if __name__ == "__main__":
    # run app
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())