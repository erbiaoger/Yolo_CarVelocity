from ultralytics import YOLO
import pathlib
from PIL import Image
import time

start = time.time()
# 加载一个模型
model = YOLO('runs/detect/train13/weights/best.pt')  # 从YAML建立一个新模型


# list_file = pathlib.Path('datasets/car_dataset/images/train').glob('*.png')
list_file = pathlib.Path('/home/lty/CUT_plus/results/car_some_data/test_latest/images/fake_B').glob('*.png')
files = [str(file) for file in list_file]

results = model(source=files, save=False, nms=True, iou=0.2)





color = (255, 193, 193)  # BGR

dir_path = pathlib.Path('output2/')


i = 0
for r in results:
    i += 1

    boxes = r.boxes
    # masks = r.masks
    # probs = r.probs
    
    dx = 280. / r.orig_img.shape[1]
    dt = 60. / r.orig_img.shape[0]

    
    try:
        cls = boxes.cls.cpu().numpy().astype(int)[0]       # 类别
        # conf = boxes.conf.cpu().numpy()                 # 置信度
        xyxy = boxes.xyxy[0].cpu().numpy().astype(int)  # 边界框
        # xywh = boxes.xywh[0].cpu().numpy().astype(int)  # 边界框
    except:
        file_name = r.path.split('/')[-1]
        im_array = r.plot(line_width=1, font_size=0.5, boxes=True, show_vel=False, dx=dx*3.6, dt=dt)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        im.save(dir_path/file_name)
        continue

    file_name = r.path.split('/')[-1]
    im_array = r.plot(line_width=1, boxes=False, show_vel=True, dx=dx*3.6, dt=dt)  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    # im.show()  # show image
    im.save(dir_path/file_name)

end = time.time()
print('time: ', end-start)
