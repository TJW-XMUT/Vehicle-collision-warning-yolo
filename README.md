# 前言
本项目效果展示视频：
视频一：
[https://www.bilibili.com/video/BV1SN41177q9/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a](https://www.bilibili.com/video/BV1SN41177q9/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a)

视频二（带QT界面）：
[https://www.bilibili.com/video/BV1wXrHYGEbm/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a](https://www.bilibili.com/video/BV1wXrHYGEbm/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a)

**可提供整套代码(含详细注释)、训练好的权重、数据集、测试视频和详细说明文档。可以部署到树莓派、香橙派、Jetson Nano、瑞芯微RK3588等开发板上，也可调用摄像头输入视频流进行实时推理。**

1、本项目通过yolov11/yolov10/yolov9/yolov8/yolov7/yolov5和deepsort实现了一个自动驾驶领域的追尾前车碰撞预警系统，可为一些同学的课设、大作业等提供参考，带QT界面。分别实现了自行车、汽车、摩托车、公交车、卡车的实时目标检测、跟车距离测量、车辆间的相对速度测量、基于人脑反应时间和车辆刹停时间的碰撞预警功能。最终效果如下，红色框代表易发生碰撞追尾的高风险目标，黄色框代表中风险目标，绿色框代表低风险目标。
2、可训练自己的数据集，可以换成yolov10/yolov9/yolov8/yolov7/yolov5各种版本的权重。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d5794e861c424f32b9a1a5687b7d50de.gif#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d7250cf58625832b16f6e272a088ef59.gif#pic_center)


# 一、项目环境配置
不熟悉pycharm的anaconda的大兄弟请先看这篇csdn博客，了解pycharm和anaconda的基本操作。
[https://blog.csdn.net/ECHOSON/article/details/117220445](https://blog.csdn.net/ECHOSON/article/details/117220445)
anaconda安装完成之后请切换到国内的源来提高下载速度 ，命令如下：

```python
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```
首先创建python3.8的虚拟环境，请在命令行中执行下列操作：

```python
conda create -n yolov8 python==3.8.5
conda activate yolov8
```
## 1、pytorch安装（gpu版本和cpu版本的安装)
实际测试情况是yolov10/yolov9/yolov8/yolov7/yolov5在CPU和GPU的情况下均可使用，不过在CPU的条件下训练那个速度会令人发指，所以有条件的小伙伴一定要安装GPU版本的Pytorch，没有条件的小伙伴最好是租服务器来使用。GPU版本安装的具体步骤可以参考这篇文章：[https://blog.csdn.net/ECHOSON/article/details/118420968](https://blog.csdn.net/ECHOSON/article/details/118420968)。
需要注意以下几点：
1、安装之前一定要先更新你的显卡驱动，去官网下载对应型号的驱动安装
2、30系显卡只能使用cuda11的版本
3、一定要创建虚拟环境，这样的话各个深度学习框架之间不发生冲突
我这里创建的是python3.8的环境，安装的Pytorch的版本是1.8.0，命令如下：

```python
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
```
安装完毕之后，我们来测试一下GPU是否可以有效调用：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a79b18bf74273e070fa9ff255a6fe311.png#pic_center)
## 2、pycocotools的安装

```python
pip install pycocotools-windows
```
## 3、pyqt5安装教程
[https://blog.csdn.net/weixin_47406082/article/details/134234802?fromshare=blogdetail&sharetype=blogdetail&sharerId=134234802&sharerefer=PC&sharesource=weixin_44944382&sharefrom=from_link](https://blog.csdn.net/weixin_47406082/article/details/134234802?fromshare=blogdetail&sharetype=blogdetail&sharerId=134234802&sharerefer=PC&sharesource=weixin_44944382&sharefrom=from_link)

## 4、其他包的安装
另外的话大家还需要安装程序其他所需的包，包括opencv，matplotlib这些包，不过这些包的安装比较简单，直接通过pip指令执行即可，我们cd到yolov8代码的目录下，直接执行下列指令即可完成包的安装。

```python
pip install -r requirements.txt
```

# 二、车辆检测、实时跟踪测速算法及代码解读
## 1、主函数各参数含义
如下代码所示，可根据自己需求更改。使用yolov10/yolov9/yolov8s/yolov7s/yolov5s.pt、yolov10/yolov9/yolov8m/yolov7m/yolov5m.pt、yolov10/yolov9/yolov8l/yolov7l/yolov5.pt、yolov10/yolov9/yolov8x/yolov7x/yolov5.pt预训练权重均可，也可以使用自己训练好的权重，本项目中调用的是yolov8s.pt。
```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/videos/test.mp4', help='source')  #  file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')    # store_true为保存视频或者图片，路径为runs/detect
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')  # 结果视频的保存路径
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
```
## 2、算法实现
使用yolov5和deepsort分别实现车辆的目标检测、跟踪，再利用检测和跟踪的结果实时计算车速。首先使用提前设定好的车辆真实宽度和检测出来的车辆像素宽度求出真实距离和像素距离的比值，再使用每辆车的前后两帧框的中心坐标计算出两帧之间移动的像素距离。利用这个比值和像素距离做映射，就可以求出两帧之间车辆移动的真实距离。然后距离除以两帧之间的时间，就是速度了。本测速算法中将车辆真实移动距离与像素移动距离看成是线性关系，仅在监控相机轴线与车辆移动方向垂直时才能成立，并且检测出来的车辆框在空间上会产生一定形变，使得真实距离和像素距离的映射关系不准确。有兴趣的同学可以在代码中加入透视变换，将图像变成类似于遥感数据的俯瞰图，实现测速后再将图像变换为原始图像视角。
## 3、核心代码
我的项目将测速代码封装到了Estimated_speed()函数里面，有详细注释，调用即可。需要注意的是，由于本项目测试视频为行车记录仪视角所拍摄，拍摄设备本身也在移动，此处测得的车速为车辆之间的相对速度。
```python
def Estimated_speed(locations, fps, width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []  # 当前帧数据：中心点x坐标、中心点y坐标、目标序号、车辆类别、车辆像素宽度
    work_prev_locations = []  # 上一帧数据，数据格式相同
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])  # 获得当前帧中跟踪到车辆的ID
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])  # 获得前一帧中跟踪到车辆的ID
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:  # 进行筛选，找到在两帧图像中均被检测到的有效车辆ID，存入work_IDs中
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:  # 将当前帧有效检测车辆的信息存入work_locations中
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:  # 将前一帧有效检测车辆的ID索引存入work_IDs_prev_index中
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:  # 将前一帧有效检测车辆的信息存入work_prev_locations中
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]) ** 2 +  # 计算有效检测车辆的速度，采用线性的从像素距离到真实空间距离的映射
                      (work_locations[i][1] - work_prev_locations[i][1]) ** 2) *  # 当视频拍摄视角并不垂直于车辆移动轨迹时，测算出来的速度将比实际速度低
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 * 3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]]  # 将保留一位小数的单位为km/h的车辆速度及其ID存入speed二维列表中
    return speed
```
另外，我的项目中将每辆车的中心坐标轨迹和车速分别写入了根目录下的track.txt和speed.txt，实现了每辆车的速度和轨迹信息记录。
```python
# 将每帧检测出来的目标中心坐标和车辆ID写入txt中,实现轨迹跟踪
if len(location) != 0:
    with open('track.txt', 'a+') as track_record:
        track_record.write('frame:%s\n' % str(frame_idx))
        for j in range(len(location)):
            track_record.write('id:%s,x:%s,y:%s\n' % (str(location[j][2]), str(location[j][0]), str(location[j][1])))
    print('done!')
locations.append(location)
print(len(locations))
# 每五帧写入一次测速的数据，进行测速
if len(locations) == 5:
    if len(locations[0]) and len(locations[-1]) != 0:
        locations = [locations[0], locations[-1]]
        speed = Estimated_speed(locations, fps, width)
    with open('speed.txt', 'a+') as speed_record:
        for sp in speed:
            speed_record.write('id:%s %skm/h\n' % (str(sp[1]), str(sp[0])))  # 将每辆车的速度写入项目根目录下的speed.txt中
    locations = []
```
## 4、效果展示
如图所示，每个目标车辆测出来的速度和行驶轨迹的中心坐标分别存储在两个txt里面，id值用于区分不同的车辆，frame值代表视频的第几帧，x、y分别表示横纵坐标值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/70ecd35315cb4b38af7966b1268c3ec7.png##)


# 三、跟车距离测量算法及代码解读
## 1、主函数各参数含义
```python
foc = 500.0        # 镜头焦距,单位为mm
real_hight_bicycle = 26.04      # 自行车高度，注意单位是英寸
real_hight_car = 59.08      # 汽车高度
real_hight_motorcycle = 47.24      # 摩托车高度
real_hight_bus = 125.98      # 公交车高度
real_hight_truck = 137.79   # 卡车高度

# 自定义函数，单目测距
def detect_distance_car(h):
    dis_inch = (real_hight_car * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/100
    return dis_m
```
## 2、算法实现
车辆距离计算公式：D = (F*W)/P，其中D是目标到摄像机的距离(即车辆距离）, Ｆ是相机焦距, W是目标的宽度或者高度, P是指目标在图像中所占据的x方向像素的宽或者y方向像素的高（由目标检测结果可获取）。首先需要设置好镜头焦距，这个参数可以通过在网上查询拍摄设备的参数获取，我这里用的测试视频使用行车记录仪拍摄，焦距为500mm，然后分别设置好自行车、汽车、摩托车、公交车和卡车的实际高度（单位为英寸），利用该公式就能计算出前车距离。本质上就是通过车辆现实尺寸和像素尺寸实现了一个距离映射。
## 3、效果展示
如图所示，1.6km/h代表这辆车相对拍摄设备行驶的相对速度，car代表目标类别为汽车，0.83为目标的置信度，2.42m为测得的跟车距离。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/344be63972fb5fabba043b9b8b5cf7cd.png)

# 四、前车碰撞预警（追尾预警）算法及代码解读
## 1、算法实现
首先通过detect.py函数里的time_person变量设置人脑反应后的刹车时间，单位为s，即人开始反应后踩下刹车到车辆刹停的时间，这个时间与车辆本身的速度有关，后续可通过车机系统接口读取该速度，实现更好的预警效果。这里我们的预设值为3s。
```python
    time_person = 3   # 设置人脑反应后的刹车时间，单位为s，即从人反应后踩下刹车到车辆刹停的时间，这个时间与车辆本身的速度有关，后续可通过车机系统接口读取该速度，实现更好的预警效果
```
再调用plot_one_box()函数，将前述变量 time_person、所测得的车辆目标速度、类别名称等值传入。

```python
plot_one_box(xyxy, im0, speed, outputs, time_person, label=label, color=[0, 0, 255], line_thickness=3, name=names[int(cls)])  # 调用函数进行不同类别的测距，并绘制目标框
```
plot_one_box()函数在plots.py中的定义如下，首先根据不同的标签名称调用不同的函数计算跟车距离，再利用测出来的速度和距离计算时间t，与预先设定的人脑反应后的刹车时间time_person在draw_speed()函数中进行比较，并返回一个标记值flag。若时间t小于time_person的1/2，则判定为高风险，并将车辆目标绘制为红色框进行预警；若时间t介于time_person和time_person的1/2之间，则判定为低风险，并将车辆目标绘制为黄色框进行预警；若时间t大于time_person，则并将车辆目标绘制为绿色框，判定为无风险。
```python
def plot_one_box(x, img, speed, outputs, time_person, color=None, label=None, line_thickness=3, name=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    # w = int(x[2]) - int(x[0])  # 框的宽
    h = int(x[3]) - int(x[1])    # 框的高
    dis_m = 1.00
    if name == 'bicycle':    # 根据标签名称调用不同函数计算距离
        dis_m = detect_distance_bicycle(h)
    elif name == 'car':
        dis_m = detect_distance_car(h)
    elif name == 'motorcycle':
        dis_m = detect_distance_motorcycle(h)
    elif name == 'bus':
        dis_m = detect_distance_bus(h)
    elif name == 'truck':
        dis_m = detect_distance_truck(h)
    label += f'  {dis_m}m'    # 在标签后追加距离
    # 利用测出来的速度和距离计算时间，与预先设定的人脑反应后的刹车时间进行比较，
    flag=''
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -2]
        img, flag = draw_speed(img, speed, bbox_xyxy, identities, time_person, dis_m)
    if flag == "High risk":   # 根据判定的不同风险等级，绘制不同颜色的目标框，起到预警的作用
        cv2.rectangle(img, c1, c2, [0, 0, 255], thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    elif flag == "Low risk":
        cv2.rectangle(img, c1, c2, [0, 215, 255], thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, [0, 215, 255], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    else:
        cv2.rectangle(img, c1, c2, [48, 128, 20], thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, [48, 128, 20], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
```
## 2、效果展示
如图所示，正前方有四辆车由于跟车距离过近和相对速度过快，触发了系统的预警功能，目标框分别显示为红色和黄色，起到对驾驶员或自动驾驶系统进行提醒的作用。还有目标由于距离过远，对车辆的行车安全不构成威胁，所以显示为绿色框。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/3ac1cbe0cc33a9220fcfe2661695cd4a.png)
# 五、总结及源码获取
## 1、总结
本项目基于深度目标检测和跟踪技术，结合了一些图像逻辑后处理算法，实现了车辆检测、跟踪、测速、车间距离的测量和前车碰撞预警的功能，检测准确率较高，算法实时性较好，对于自动驾驶车辆的交通安全和环境感知具有一定参考意义和实用价值。
## 2、项目资源获取(yolov11/yolov10/yolov9/yolov8/yolov7/yolov5版本均可提供)
**可提供整套代码加训练好的权重，还有测试视频和详细说明文档。代码有详细注释，包全程指导，任何问题都可以随时问我。不过有的时候我太忙，可能不会及时回复消息，看到了肯定回你哈**

**项目内容：**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/61f4760e6f71688924d919e1d806eb7c.png)
**包含完整word版本说明文档，可用于写论文、课设报告的参考。**
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/7ab68cc9b0ef889755d1138c9558f8fb.png)
**资源获取：**
```python
获取整套代码、测试视频、训练好的权重和说明文档(有偿)
上交硕士，技术够硬，也可以指导深度学习毕设、大作业等。
--------------->qq------------
           3582584734
------------------------------
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c7e9309a04bd6f22ae3f1138149f65ea.png)






