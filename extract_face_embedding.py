import warnings
# 忽略FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)
# 忽略UserWarning
warnings.filterwarnings('ignore', category=UserWarning)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import cv2
import numpy as np
import insightface
import traceback
from loguru import logger
from PIL import Image
from collections import UserDict
from insightface.app import FaceAnalysis
from diffusers.utils import load_image
from anime_face_detector import create_detector

def resize_img(input_image, max_side=1280, min_side=1024, size=None,
            pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image
# 图像填充
def pad_image(image):
    # 获取原始图像尺寸
    width, height = image.size

    # 如果图像尺寸小于1024，则填充至1024x1024
    if max(width, height) < 1024:
        new_size = 1024
    else:
        # 否则填充到原尺寸的2倍
        new_size = max(width, height) * 2

    # 创建一个新的白色背景图像
    new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    
    # 计算中心位置，将原图粘贴到新图中央
    left = (new_size - width) // 2
    top = (new_size - height) // 2
    new_image.paste(image, (left, top))
    return new_image
def prepare_average_embedding(face_path):
    success = False
    face_embeddings = []
    try:
        face_image = load_image(face_path)
        face_image = resize_img(face_image)
        face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
        if face_info:
            face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
            face_emb = face_info['embedding']
            face_embeddings.append(face_emb)
            success = True
        else:
            # 检测不到人脸，进行图像填充操作
            face_image = load_image(face_path)
            face_image = pad_image(face_image)
            face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
            if face_info:
                face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1]
                face_emb = face_info['embedding']
                face_embeddings.append(face_emb)
                success = True  
            # 用anime-face-detector
            else:
                face_image = load_image(face_path)
                face_image_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
                detector = create_detector('yolov3')
                face_image_mmcv = np.float32(face_image)
                preds = detector(face_image_mmcv)
                # 检测到的人脸框
                if preds and len(preds) > 0:
                    # 获取最大人脸框
                    best_box = preds[0]['bbox']  # 获取bbox字段
                    left, top, right, bottom, confidence = map(int, best_box)
                    # 裁剪出最佳人脸区域
                    face_crop = face_image_cv2[top:bottom, left:right]
                    # 将原始 landmark 坐标转换为 face_crop 的坐标
                    keypoints = preds[0]['keypoints'][:, :2]  # 获取关键点坐标 (x, y)
                    # logger.info(f"face_landmarks={keypoints}")
                    # 将landmark调整为face_crop对应坐标
                    keypoints[:, 0] -= left  
                    keypoints[:, 1] -= top   
                    # 提取鼻子和嘴的关键点
                    nose = keypoints[23]
                    mouth_left = keypoints[24]
                    mouth_right = keypoints[26]
                    # 计算左眼和右眼的中点
                    # 左眼对应的关键点索引：12, 13, 14, 15
                    left_eye_points = keypoints[[12, 13, 14, 15]]
                    left_eye_center = np.mean(left_eye_points, axis=0)

                    # 右眼对应的关键点索引：17, 18, 19, 21
                    right_eye_points = keypoints[[17, 18, 19, 21]]
                    right_eye_center = np.mean(right_eye_points, axis=0)
                    
                    class FaceWithLmk(UserDict):
                        def __init__(self, kps):
                            super().__init__()
                            self.kps = kps  # 将 bbox 设置为一个属性

                    # 从 face_lmk 中提取特定的关键点
                    selected_kps = [left_eye_center, right_eye_center, nose, mouth_left, mouth_right]
                    # logger.info(f"selected_kps={selected_kps}")
                    # 可视化kps
                    face_crop_cp = face_crop
                    for i, (x, y) in enumerate(selected_kps):
                        color = (0, 0, 255)  # 默认颜色为红色
                        if i == 0 or i == 3:  # 使第一个和第四个关键点颜色为绿色
                            color = (0, 255, 0)
                        x, y = int(x), int(y)
                        cv2.circle(face_crop_cp, (int(x), int(y)), radius=5, color=color, thickness=-1)
                        cv2.putText(face_crop_cp, str(i), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # 保存最佳人脸图像
                    cv2.imwrite(f'./kps_{face_path.split("/")[-1].split(".")[0]}.jpg', face_crop_cp)
                    # 创建包含选定关键点的 FaceWithLmk 实例
                    face_with_lmk = FaceWithLmk(kps=np.array(selected_kps))
                    id_ante_embedding = embedding_ante.get(face_crop, face_with_lmk)
                    face_embeddings.append(id_ante_embedding)
                    success = True 
    except Exception as e:
        # 捕获详细的堆栈信息
        error_msg = traceback.format_exc()  # 获取完整的堆栈信息
        logger.error(f"Prepare face embedding {face_path} error: {e}\n{error_msg}")
    return face_embeddings, success                
    
    

if __name__ == '__main__':
    app = FaceAnalysis(name="antelopev2", root='./', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    embedding_ante = insightface.model_zoo.get_model('models/antelopev2/glintr100.onnx')
    embedding_ante.prepare(ctx_id=0)

    image_path = "/path/to/yourimage"
    image_path = "/mnt/data/kaijia.yan/github_for_me/face_embedding_extractor/assets/test/real_dilireba.png"
    face_emb, success = prepare_average_embedding(image_path)
