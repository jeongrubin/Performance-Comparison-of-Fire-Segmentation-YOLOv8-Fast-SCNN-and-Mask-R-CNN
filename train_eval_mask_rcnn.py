#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 데이터셋 클래스 정의
class FireMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '.png'))

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # 마스크를 NumPy 배열로 변환하고 0과 1로 변환
        mask = np.array(mask)  # NumPy 배열로 변환
        mask = (mask > 0).astype(np.float32)  # 마스크 값이 1인 부분은 1로 설정하고, 나머지는 0으로 설정
        
        # NumPy 배열을 다시 텐서로 변환
        mask = torch.from_numpy(mask)

        # 변환 적용 (이미지와 마스크 모두)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# 데이터 로드 및 전처리
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'

transform = transforms.Compose([
    transforms.ToTensor()  # 이미지를 텐서로 변환
])

train_dataset = FireMaskDataset(train_image_dir, train_mask_dir, transform=transform)
val_dataset = FireMaskDataset(val_image_dir, val_mask_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# 모델 정의
def get_mask_rcnn_model(num_classes):
    # FPN을 사용하여 ResNet 백본 정의
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model

# 모델 초기화
num_classes = 2  # 배경과 불 마스크 두 개의 클래스
model = get_mask_rcnn_model(num_classes)
model.train()

# 모델 훈련 함수
def train_model(model, dataloader, num_epochs):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(num_epochs):
        total_loss = 0
        for images, masks in dataloader:
            images = [image.to(device) for image in images]
            masks = [mask.to(device) for mask in masks]

            # 각 배치의 타겟 설정
            targets = []
            for mask in masks:
                target = {}
                target['masks'] = mask.unsqueeze(0)  # 채널 차원 추가
                target['boxes'] = torch.tensor([[0, 0, mask.shape[1], mask.shape[2]]], dtype=torch.float32)  # 예시 박스 (전체 이미지)
                target['labels'] = torch.tensor([1])  # 불 마스크 클래스 레이블
                targets.append(target)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(dataloader)}')

# 모델 훈련
num_epochs = 10
train_model(model, train_loader, num_epochs)


# In[6]:


import os
import matplotlib.pyplot as plt
from PIL import Image

# 이미지 디렉토리 경로 설정
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'

# 특정 개수의 이미지를 출력하는 함수 정의
def display_images(image_dir, num_images=5):
    image_files = os.listdir(image_dir)[:num_images]  # 지정한 개수만큼 이미지 파일 목록을 가져옴
    plt.figure(figsize=(15, 10))  # 출력할 이미지 크기 설정

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)  # 이미지 열기
        plt.subplot(1, num_images, i + 1)  # 서브플롯 설정
        plt.imshow(image)  # 이미지 출력
        plt.axis('off')  # 축 숨기기
        plt.title(image_file)  # 이미지 파일 이름 제목으로 설정

    plt.show()  # 모든 이미지를 화면에 출력

# 훈련 이미지 출력
display_images(train_image_dir, num_images=5)

# 검증 이미지 출력
display_images(val_image_dir, num_images=5)


# In[16]:


import os
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Dataset Class
def load_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '.png'))
        image, mask = load_image_and_mask(image_path, mask_path)

        # Convert to tensors
        image = F.to_tensor(image)
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        # Create target dictionary
        obj_ids = torch.unique(mask)[1:]  # Exclude background
        masks = mask == obj_ids[:, None, None]

        boxes = []
        for m in masks:
            pos = torch.where(m)
            xmin, ymin, xmax, ymax = pos[1].min(), pos[0].min(), pos[1].max(), pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)  # All foreground

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

# Paths
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'

# Dataset and DataLoader
train_dataset = CustomDataset(train_image_dir, train_mask_dir)
val_dataset = CustomDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.train()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

# Training Loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")


# In[17]:


import os
import cv2
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Dataset Class
def load_image_and_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return image, mask

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.image_filenames[idx].replace('.jpg', '.png'))
        image, mask = load_image_and_mask(image_path, mask_path)

        # Convert to tensors
        image = F.to_tensor(image)
        mask = torch.as_tensor(mask, dtype=torch.uint8)

        # Create target dictionary
        obj_ids = torch.unique(mask)[1:]  # Exclude background
        masks = mask == obj_ids[:, None, None]

        boxes = []
        for m in masks:
            pos = torch.where(m)
            xmin, ymin, xmax, ymax = pos[1].min(), pos[0].min(), pos[1].max(), pos[0].max()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)  # All foreground

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

# Paths
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'

# Dataset and DataLoader
train_dataset = CustomDataset(train_image_dir, train_mask_dir)
val_dataset = CustomDataset(val_image_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.train()

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, targets in tqdm(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

# Training Loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")


# In[19]:


# 학습 완료된 모델 저장
save_path = "mask_rcnn_model.pth"
torch.save(model.state_dict(), save_path)
print(f"Trained model saved to {save_path}")


# In[20]:


# 저장된 모델 로드
model = maskrcnn_resnet50_fpn(pretrained=False)  # 사전 학습된 가중치 불필요
model.load_state_dict(torch.load("mask_rcnn_model.pth"))
model.to(device)
model.eval()  # 평가 모드로 전환


# In[21]:


import os
import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("mask_rcnn_model.pth"))
model.to(device)
model.eval()

# 테스트 이미지 디렉토리
test_image_dir = "/home/wuyt8807/Fast-SCNN-pytorch/data/test/images"
output_dir = "/home/wuyt8807/Fast-SCNN-pytorch/data/test/output"
os.makedirs(output_dir, exist_ok=True)

# 테스트 함수
def predict_and_visualize(model, image_path, output_path, threshold=0.5):
    # 이미지 로드
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 마스크와 박스 시각화
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    
    # 불 부분 필터링 (스코어 기준)
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # 마스크 추출
            box = boxes[i].cpu().numpy().astype(int)          # 박스 추출
            xmin, ymin, xmax, ymax = box
            
            # 이미지에 마스크 덧씌우기
            image[mask > 128] = [0, 0, 255]  # 빨간색으로 불 표시
            # 경계 박스 그리기
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # 파란색 박스
    
    # 결과 저장
    output_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Saved output to {output_path}")

# 테스트 실행
for image_file in os.listdir(test_image_dir):
    image_path = os.path.join(test_image_dir, image_file)
    predict_and_visualize(model, image_path, output_dir)

print("Test complete.")


# In[23]:


import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import os
from PIL import Image
import random

# 모델 로드
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load('mask_rcnn_model.pth'))
model.eval()

# 이미지 변환
transform = T.Compose([
    T.ToTensor(),
])

# 이미지 로드 및 예측
def load_images_from_folder(folder, num_images=20):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            images.append(img_path)
    return random.sample(images, num_images)

def predict_and_plot(images):
    fig, axs = plt.subplots(len(images), 2, figsize=(10, 5 * len(images)))

    for i, img_path in enumerate(images):
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = model(image_tensor)

        # 예측 결과에서 불 객체만 필터링 (예: 클래스 ID가 1인 경우)
        masks = prediction[0]['masks'] > 0.5
        pred_mask = masks.sum(dim=0).byte().cpu().numpy()

        # 원본 이미지와 예측 결과 표시
        axs[i, 0].imshow(image)
        axs[i, 0].set_title('Original Image')
        axs[i, 0].axis('off')

        # pred_mask를 2차원 배열로 변환
        if len(pred_mask.shape) > 2:
            pred_mask = pred_mask[0]  # 첫 번째 마스크만 사용

        axs[i, 1].imshow(image)
        axs[i, 1].imshow(pred_mask, alpha=0.5, cmap='jet')
        axs[i, 1].set_title('Predicted Segmentation')
        axs[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


# 테스트 이미지 로드
test_images = load_images_from_folder('/home/wuyt8807/Fast-SCNN-pytorch/data/test/images', num_images=20)
predict_and_plot(test_images)


# In[25]:


import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

# 모델 로드
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load('mask_rcnn_model.pth'))
model.eval()

# 이미지 변환
transform = T.Compose([
    T.ToTensor(),
])

# 비디오 파일 경로
video_path = '/home/wuyt8807/Fast-SCNN-pytorch/화재2.mp4'

# 비디오 캡처
cap = cv2.VideoCapture(video_path)

# FPS 계산을 위한 변수 초기화
fps_list = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    start_time = time.time()

    # OpenCV에서 BGR 형식으로 읽은 프레임을 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0)

    with torch.no_grad():
        prediction = model(image_tensor)

    # 예측 결과에서 불 객체만 필터링 (예: 클래스 ID가 1인 경우)
    masks = prediction[0]['masks'] > 0.5
    pred_mask = masks.sum(dim=0).byte().cpu().numpy()

    # FPS 계산
    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    fps_list.append(fps)

    # 결과 표시
    plt.imshow(image)
    
    # pred_mask를 2차원 배열로 변환
    if len(pred_mask.shape) > 2:
        pred_mask = pred_mask[0]  # 첫 번째 마스크만 사용

    plt.imshow(pred_mask, alpha=0.5, cmap='jet')
    plt.title(f'Frame {frame_count} - FPS: {fps:.2f}')
    plt.axis('off')
    plt.show()

# 비디오 캡처 종료
cap.release()

# 평균 FPS 출력
average_fps = np.mean(fps_list)
print(f'Average FPS: {average_fps:.2f}')


# In[26]:


import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("/home/wuyt8807/Fast-SCNN-pytorch/mask_rcnn_model.pth"))
model.to(device)
model.eval()

# 입력 동영상과 출력 동영상 경로
input_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2.mp4"
output_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2_predicted.mp4"

# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def predict_and_visualize_frame(model, frame, threshold=0.5):
    # 이미지 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 마스크와 박스 시각화
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    prediction_frame = frame.copy()
    
    # 불 부분 필터링 (스코어 기준)
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # 마스크 추출
            box = boxes[i].cpu().numpy().astype(int)          # 박스 추출
            xmin, ymin, xmax, ymax = box
            
            # 마스크 덧씌우기
            prediction_frame[mask > 128] = [0, 0, 255]  # 빨간색
            # 경계 박스 그리기
            cv2.rectangle(prediction_frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)  # 하얀색 박스
    
    return prediction_frame

# 프레임별 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 예측 및 시각화
    predicted_frame = predict_and_visualize_frame(model, frame)
    # 동영상 저장
    out.write(predicted_frame)

# 리소스 해제
cap.release()
out.release()
print(f"Predicted video saved to {output_video_path}")


# In[ ]:


import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("/home/wuyt8807/Fast-SCNN-pytorch/mask_rcnn_model.pth"))
model.to(device)
model.eval()

# 입력 동영상과 출력 동영상 경로
input_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2.mp4"
output_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2_predicted.mp4"

# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def predict_and_visualize_frame(model, frame, threshold=0.5):
    # 이미지 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 마스크와 박스 시각화
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    prediction_frame = frame.copy()
    
    # 불 부분 필터링 (스코어 기준)
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # 마스크 추출
            box = boxes[i].cpu().numpy().astype(int)          # 박스 추출
            xmin, ymin, xmax, ymax = box
            
            # 마스크 덧씌우기
            prediction_frame[mask > 128] = [0, 0, 255]  # 빨간색
            # 경계 박스 그리기
            cv2.rectangle(prediction_frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)  # 하얀색 박스
    
    return prediction_frame

# 프레임별 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 예측 및 시각화
    predicted_frame = predict_and_visualize_frame(model, frame)
    # 동영상 저장
    out.write(predicted_frame)

# 리소스 해제
cap.release()
out.release()
print(f"Predicted video saved to {output_video_path}")


# In[ ]:


import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("/home/wuyt8807/Fast-SCNN-pytorch/mask_rcnn_model.pth"))
model.to(device)
model.eval()

# 입력 동영상과 출력 동영상 경로
input_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2.mp4"
output_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2_predicted.mp4"

# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def predict_and_visualize_frame(model, frame, threshold=0.5):
    # 이미지 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 마스크와 박스 시각화
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    prediction_frame = frame.copy()
    
    # 불 부분 필터링 (스코어 기준)
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # 마스크 추출
            box = boxes[i].cpu().numpy().astype(int)          # 박스 추출
            xmin, ymin, xmax, ymax = box
            
            # 마스크 덧씌우기
            prediction_frame[mask > 128] = [0, 0, 255]  # 빨간색
            # 경계 박스 그리기
            cv2.rectangle(prediction_frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)  # 하얀색 박스
    
    return prediction_frame

# 프레임별 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 예측 및 시각화
    predicted_frame = predict_and_visualize_frame(model, frame)
    # 동영상 저장
    out.write(predicted_frame)

# 리소스 해제
cap.release()
out.release()
print(f"Predicted video saved to {output_video_path}")


# In[28]:


import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("/home/wuyt8807/Fast-SCNN-pytorch/mask_rcnn_model.pth"))
model.to(device)
model.eval()

# 입력 동영상과 출력 동영상 경로
input_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/불불.mp4"
output_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/불불2_predicted.mp4"

# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

def predict_and_visualize_frame(model, frame, threshold=0.5):
    # 이미지 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 마스크와 박스 시각화
    masks = predictions[0]['masks']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    prediction_frame = frame.copy()
    
    # 불 부분 필터링 (스코어 기준)
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = masks[i, 0].mul(255).byte().cpu().numpy()  # 마스크 추출
            box = boxes[i].cpu().numpy().astype(int)          # 박스 추출
            xmin, ymin, xmax, ymax = box
            
            # 마스크 덧씌우기
            prediction_frame[mask > 128] = [0, 0, 255]  # 빨간색
            # 경계 박스 그리기
            cv2.rectangle(prediction_frame, (xmin, ymin), (xmax, ymax), (255, 255, 255), 2)  # 하얀색 박스
    
    return prediction_frame

# 프레임별 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 예측 및 시각화
    predicted_frame = predict_and_visualize_frame(model, frame)
    # 동영상 저장
    out.write(predicted_frame)

# 리소스 해제
cap.release()
out.release()
print(f"Predicted video saved to {output_video_path}")


# In[37]:


import cv2
import time
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from sklearn.metrics import precision_recall_fscore_support

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load("/home/wuyt8807/Fast-SCNN-pytorch/mask_rcnn_model.pth"))
model.to(device)
model.eval()

# 입력 동영상과 출력 동영상 경로
input_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2.mp4"
output_video_path = "/home/wuyt8807/Fast-SCNN-pytorch/화재2예측.mp4"

# 동영상 읽기
cap = cv2.VideoCapture(input_video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# FPS 측정
frame_count = 0
start_time = time.time()

# F1-Score 계산을 위한 변수
true_labels = []  # 실제 불이 있는 영역 (ground truth)
predicted_labels = []  # 모델이 예측한 영역
no_detection_frames = 0  # 감지되지 않은 프레임 수

def predict_and_evaluate_frame(model, frame, ground_truth_mask=None, threshold=0.3):
    global true_labels, predicted_labels, no_detection_frames

    # 이미지 변환
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # 모델이 아무 객체도 감지하지 못한 경우
    if len(predictions[0]['masks']) == 0:
        no_detection_frames += 1
        return frame  # 원본 프레임 그대로 반환

    # 마스크와 박스 시각화
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']
    predicted_frame = frame.copy()
    
    # 예측된 불 부분 필터링 (스코어 기준)
    predicted_mask = torch.zeros_like(masks[0][0], dtype=torch.uint8)  # 빈 마스크
    for i in range(len(masks)):
        if scores[i] > threshold:
            predicted_mask |= (masks[i, 0] > 0.5).byte()  # 예측 마스크 업데이트
    
    # 평가를 위해 True/False 값 추가
    if ground_truth_mask is not None:
        true_labels.append(ground_truth_mask.flatten())
        predicted_labels.append(predicted_mask.flatten().cpu().numpy())

    return predicted_frame

# 동영상 프레임별 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    # 예측
    predicted_frame = predict_and_evaluate_frame(model, frame, threshold=0.1)
    # 저장
    out.write(predicted_frame)

# FPS 계산
end_time = time.time()
fps = frame_count / (end_time - start_time)

# F1-Score 계산
if true_labels and predicted_labels:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=np.concatenate(true_labels), 
        y_pred=np.concatenate(predicted_labels), 
        average='binary'
    )
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
else:
    print("No objects detected in the entire video. Unable to calculate F1-Score.")

# 리소스 해제
cap.release()
out.release()

# 결과 출력
print(f"FPS: {fps:.2f}")
print(f"No detection frames: {no_detection_frames}/{frame_count} frames.")


# In[36]:


print(predictions[0])


# In[32]:


# F1-Score 계산
if true_labels and predicted_labels:  # 두 리스트가 비어있지 않은 경우에만 계산
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=np.concatenate(true_labels), 
        y_pred=np.concatenate(predicted_labels), 
        average='binary'
    )
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
else:
    print("No objects detected in the entire video. Unable to calculate F1-Score.")

