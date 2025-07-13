

import cv2
import numpy as np

# 填充灰条，实现resize
def letterbox_image(image, size):
    ih, iw, _   = np.shape(image)
    w, h        = size
    scale       = min(w/iw, h/ih)
    nw          = int(iw*scale)
    nh          = int(ih*scale)

    image       = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3]) * 128
    new_image[(h-nh)//2:nh+(h-nh)//2, (w-nw)//2:nw+(w-nw)//2] = image
    return new_image

# 后处理，暂不需要
def draw_bbox(frame, dets):
    for b in dets:
        if b[4] < 0.9:
            continue
        # map函数：将int()函数作用在b的每个元素上
        x1, y1, x2, y2, _ = map(int, b)
        # 保证框不越界
        x1 = max(0, min(x1, frame.shape[1]))    # x1为框的左上角x坐标
        y1 = max(0, min(y1, frame.shape[0]))
        x2 = max(0, min(x2, frame.shape[1]))
        y2 = max(0, min(y2, frame.shape[0]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return frame

#解码操作
def decode(loc, priors, variances):
    boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                    priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

# 关键点解码
def decode_landmks(pre, priors, variances):
    landmks = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), 1)
    return landmks