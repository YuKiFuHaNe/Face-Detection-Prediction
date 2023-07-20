import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector   #导入检测器

## 下面是导入摄像头
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces = 1)  #检测人脸

while True:
    success,img = cap.read()
    img,faces = detector.findFaceMesh(img,draw = True
                                      )    #做检测器，查找面部网格，返回图像（img）和我们的面部（faces）  ,draw = False有了这句话后就看不到网格了

    if faces:      #如果有面部（faces）可用
        ##通过以下语句找到左眼和右眼两个点
        face = faces[0]   #先进行第一阶段
        pointLeft = face[145]     #左边的值基本上是值145
        pointRight = face[374]    #右边的值基本上是值374
        ##下面是找眼睛两个点之间的距离
        # cv2.line(img, pointLeft, pointRight, (0, 200, 0), 3)  # 在眼睛两个点之间画一条线，起点是pointLeft,终点是pointRight,线条颜色为绿色，线宽为3
        # cv2.circle(img,pointLeft,5,(255,0,255),cv2.FILLED)     #在img图像上画圆，中心点为pointLeft,半径为5，颜色为紫色，最后的运行结果能够在成像人的左眼标出紫色的圆点
        # cv2.circle(img,pointRight,5,(255,0,255),cv2.FILLED)    #在img图像上画圆，中心点为pointRighe，半径为5，颜色为紫色，最后的运行结果能够在成像人的右眼标出紫色的圆点

        w, _ = detector.findDistance(pointLeft, pointRight)  # 将左眼点的位置到右眼点位置的距离赋值给w        w后面的下划线是忽略其他的值
        W = 6.3  # 这是人眼的左眼与右眼之间的距离为63mm，取了中间值，男人的为64mm，女人的为62mm

        ###查找距离
        ##通过上面f = (w * d) / W公式，可以大致的测出，当人眼距离相机50cm时，相机的焦距为300左右
        ##再将找到的焦距代入计算距离的公式，就可以算出距离
        f = 300
        d = (W * f) / w
        print(d)

        ##下面是将距离的文本字样跟随人脸移动输出在额头位置
        cvzone.putTextRect(img,f'Depth:{int(d)}cm',(face[10][0]-95,face[10][1]-5),scale = 1.8)   #将距离文本显示在图像上，以字符串形式显示，单位为cm,文本值显示的位置跟随人面部移动显示在额头上（额头上的id是10，也就是face[10],后面的face[10][0]表示第一个元素，face[10][1]表示第二个元素），
        ##上面scale = 2表示输出文本框在图片上的大小
        ##face[10][0]改变的是左右，face[10][1]改变的是显示的高度

    cv2.imshow("Iamge",img)
    cv2.waitKey(1)


