import lcd
import sensor
import image
import gc
from maix import KPU
from board import board_info
from fpioa_manager import fm
from maix import GPIO
import time
from machine import UART

# 初始化蜂鸣器
fm.register(board_info.BEEP, fm.fpioa.GPIO2)
beep = GPIO(GPIO.GPIO2, GPIO.OUT)
beep.value(0)  # 初始关闭蜂鸣器

# 初始化串口
fm.register(35, fm.fpioa.UART2_TX, force=True)
fm.register(34, fm.fpioa.UART2_RX, force=True)
uart = UART(UART.UART2, 115200, 8, None, 1, timeout=1000, read_buf_len=4096)

# 初始化摄像头和LCD
lcd.init()
sensor.reset()
sensor.set_framesize(sensor.QVGA)
sensor.set_pixformat(sensor.RGB565)
sensor.set_vflip(True)

# YOLOv2配置
resize_img = image.Image(size=(320, 256))
anchor = (1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071)
names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", 
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 目标类别及其对应的ID和编号
target_classes_info = {
    "car": 1,
    "bus": 2, 
    "motorbike": 3,
    "bicycle": 4,
    "cat": 5,
    "dog": 6,
    "person": 7
}

target_ids = [names.index(name) for name in target_classes_info.keys()]

# 构造KPU对象
object_detecter = KPU()
# 加载模型文件
object_detecter.load_kmodel("/sd/KPU/voc20_detect.kmodel")
# 初始化YOLO2网络
object_detecter.init_yolo2(anchor, anchor_num=len(anchor) // 2, img_w=320, img_h=240, 
                          net_w=320, net_h=256, layer_w=10, layer_h=8, 
                          threshold=0.5, nms_value=0.2, classes=len(names))

# 控制变量
last_action_time = 0
beep_duration = 50  # 蜂鸣器响的持续时间(毫秒)
action_interval = 1000  # 动作触发的最小间隔(毫秒)
last_detected_classes = set()  # 记录上次检测到的类别

while True:
    img = sensor.snapshot()
    resize_img.draw_image(img, 0, 0).pix_to_ai()
    
    # 进行KPU运算
    object_detecter.run_with_output(resize_img)
    # 进行YOLO2运算
    objects = object_detecter.regionlayer_yolo2()
    
    # 检测目标物体
    current_time = time.ticks_ms()
    target_detected = False
    current_detected_classes = set()  # 当前帧检测到的类别
    
    for obj in objects:
        # 绘制检测结果
        img.draw_rectangle(obj[0], obj[1], obj[2], obj[3], color=(0, 255, 0))
        img.draw_string(obj[0] + 2, obj[1] + 2, "%.2f" % (obj[5]), color=(0, 255, 0))
        img.draw_string(obj[0] + 2, obj[1] + 10, names[obj[4]], color=(0, 255, 0))
        
        # 检查是否是目标类别
        if obj[4] in target_ids:
            target_detected = True
            class_name = names[obj[4]]
            current_detected_classes.add(class_name)
    
    # 控制蜂鸣器和串口发送
    if target_detected and (current_time - last_action_time > action_interval):
        # 触发蜂鸣器
        beep.value(1)
        last_action_time = current_time
        
        # 发送串口数据
        for class_name in current_detected_classes:
            if class_name in target_classes_info:
                number = target_classes_info[class_name]
                uart.write(str(number) + "/n")
                print("检测到 {}，蜂鸣器响，串口发送: {}/n".format(class_name, number))
    
    # 关闭蜂鸣器
    elif current_time - last_action_time > beep_duration:
        beep.value(0)
    
    # 更新检测记录
    last_detected_classes = current_detected_classes.copy()
    
    lcd.display(img)
    gc.collect()