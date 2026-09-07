#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow.compat.v1 as tf1
# import tensorflow as tf2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.03
session = tf1.Session(config=config)


# In[9]:


from models.fast_scnn2 import get_fast_scnn
model = get_fast_scnn(num_classes=2)
print(model)


# In[ ]:


import tensorflow.compat.v1 as tf1
import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

# TensorFlow 설정 (GPU 메모리 제한)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.03
session = tf1.Session(config=config)

# PyTorch device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 모델, 데이터 로더, 손실 함수 및 옵티마이저 설정
model = get_fast_scnn(num_classes=2).to(device)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 데이터 로더 설정 (train_dataset과 val_dataset이 이미 정의되어 있어야 함)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 훈련 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)[0]  # 모델 출력에서 메인 분류 결과 사용
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    # 검증
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images)[0]
            val_loss += criterion(val_outputs, val_masks).item()

        print(f"Validation Loss: {val_loss/len(val_loader)}")


# In[17]:


from data_loader.fire_dataset import FireDataset  # FireDataset 클래스 import

if __name__ == "__main__":
    data_root = "/home/wuyt8807/Fast-SCNN-pytorch/data"
    train_dataset = FireDataset(root=data_root, split='train')
    val_dataset = FireDataset(root=data_root, split='val')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # 데이터 샘플 확인
    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")


# In[20]:


import os
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from data_loader.fire_dataset import FireDataset
from models.fast_scnn2 import get_fast_scnn
from torch.cuda.amp import GradScaler, autocast  # Mixed Precision Training

# CustomTransform 클래스 정의
class CustomTransform:
    def __init__(self, image_transform=None, mask_transform=None):
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return {'image': image, 'mask': mask}

# PyTorch device 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# GPU 메모리 제한 설정
torch.cuda.set_per_process_memory_fraction(0.3, 0)  # GPU 메모리의 30%만 사용
torch.cuda.empty_cache()  # 캐시 초기화

# 데이터셋 경로
data_root = "/home/wuyt8807/Fast-SCNN-pytorch/data"

# 데이터 전처리: 이미지와 마스크 각각에 대해 별도 변환 적용
image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),  # 512x512로 크기 축소
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),  # 마스크 크기도 동일하게 축소
    transforms.ToTensor()
])

# CustomTransform을 사용하여 변환 설정
transform = CustomTransform(image_transform=image_transform, mask_transform=mask_transform)

# 데이터셋 로드
train_dataset = FireDataset(root=data_root, split='train', transform=transform)
val_dataset = FireDataset(root=data_root, split='val', transform=transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # 배치 크기 축소
val_loader = DataLoader(val_dataset, batch_size=2)

# 모델, 손실 함수 및 옵티마이저 설정
model = get_fast_scnn(num_classes=2).to(device)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mixed Precision Training
scaler = GradScaler()

# 학습 설정
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 20)

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # 이전 그래디언트 초기화
        with autocast():  # Mixed Precision 활성화
            outputs = model(images)[0]
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # 로그 출력 (매 10번째 배치마다)
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}: Loss: {loss.item():.4f}")

    # 에폭별 평균 손실 출력
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

    # 검증
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            with autocast():  # Mixed Precision 활성화
                val_outputs = model(val_images)[0]
                loss = criterion(val_outputs, val_masks)
                val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "/home/wuyt8807/Fast-SCNN-pytorch/weights/fire_fast_scnn.pth")
print("Model saved!")


# In[39]:


class FireTransform:
    """Custom Transform for FireDataset."""
    def __init__(self, resize=(512, 512)):
        self.resize = resize
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize),
            transforms.ToTensor()
        ])

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # Transform image
        image = self.image_transform(image)

        # Transform mask
        mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()

        return {'image': image, 'mask': mask}


# In[79]:


# CustomTransform을 사용하여 변환 설정
transform = CustomTransform(image_transform=image_transform, mask_transform=mask_transform)

# 데이터셋 로드
train_dataset = FireDataset(root=data_root, split='train', transform=transform)
val_dataset = FireDataset(root=data_root, split='val', transform=transform)

# 데이터 로더 설정
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)


# In[83]:


import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from models.fast_scnn2 import get_fast_scnn

# Custom dataset class to load images and masks
class FireDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load the image and the mask
        image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Direct paths to your dataset
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'

# Create datasets
train_dataset = FireDataset(train_image_dir, train_mask_dir, transform)
val_dataset = FireDataset(val_image_dir, val_mask_dir, transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)


# In[94]:


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import os
import numpy as np

# Custom dataset class to load images and masks
class FireDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Load the image and the mask
        image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
        mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Direct paths to your dataset
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'


# Create datasets
train_dataset = FireDataset(train_image_dir, train_mask_dir, transform)
val_dataset = FireDataset(val_image_dir, val_mask_dir, transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, criterion, and optimizer
model = get_fast_scnn(num_classes=2).to(device)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mixed Precision Training
scaler = GradScaler()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")

    # Training step
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # Clear the previous gradients

        with autocast():  # Enable mixed precision
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)  # Compute the loss

        # Backward pass and optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Print loss for every 10th batch
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}: Loss: {loss.item():.4f}")

    # Average loss for the epoch
    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            with autocast():  # Enable mixed precision
                val_outputs = model(val_images)
                loss = criterion(val_outputs, val_masks)
                val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# Save the model
torch.save(model.state_dict(), "/path/to/save/fire_fast_scnn.pth")
print("Model saved!")


# In[89]:


class FastSCNN(nn.Module):
    """Fast-SCNN"""
    def __init__(self, num_classes):
        super(FastSCNN, self).__init__()
        self.learning_to_downsample = _DSConv(3, 64, stride=2)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128)
        self.feature_fusion = _DSConv(128, 128)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        print(f"Input shape: {x.shape}")  # 입력 텐서 크기 확인
        size = x.size()[2:]
        x = self.learning_to_downsample(x)
        print(f"After learning_to_downsample shape: {x.shape}")  # 첫 번째 DSConv 후 텐서 크기
        x = self.global_feature_extractor(x)
        print(f"After global_feature_extractor shape: {x.shape}")  # global_feature_extractor 후 텐서 크기
        x = self.feature_fusion(x)
        print(f"After feature_fusion shape: {x.shape}")  # feature_fusion 후 텐서 크기
        return F.interpolate(self.classifier(x), size, mode='bilinear', align_corners=False)


# In[103]:


import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['FastSCNN', 'get_fast_scnn']


class FastSCNN(nn.Module):
    def __init__(self, num_classes, aux=False, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1)
            )

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            _ConvBNReLU(in_channels, in_channels * t, 1),
            # dw
            _DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    """Learning to downsample module"""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global feature extractor module"""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3), **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature fusion module"""

    def __init__(self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, scale_factor=4, mode='bilinear', align_corners=True)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    """Classifer"""

    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x

def get_fast_scnn(num_classes=2, **kwargs):
    return FastSCNN(num_classes=num_classes, **kwargs)


# In[96]:


from PIL import Image
import os

train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'

# 예시로 첫 번째 이미지의 차원을 확인
img_path = os.path.join(train_image_dir, os.listdir(train_image_dir)[0])
image = Image.open(img_path)

# 이미지 차원 확인 (채널, 높이, 너비)
print(f"Image size: {image.size}")  # (width, height)
print(f"Image mode: {image.mode}")  # RGB인지 확인 (3채널이 맞는지 확인)

# 이미지를 텐서로 변환하여 차원 확인
import torchvision.transforms as transforms
transform = transforms.ToTensor()
image_tensor = transform(image)

print(f"Tensor shape: {image_tensor.shape}")  # [C, H, W] 형태로 출력됨


# In[97]:


from PIL import Image
import os
import torchvision.transforms as transforms

train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'

# 예시로 첫 번째 이미지의 차원을 확인
img_path = os.path.join(train_image_dir, os.listdir(train_image_dir)[0])
image = Image.open(img_path)

# 이미지 리사이즈 (512x512)
image_resized = image.resize((512, 512))

# 이미지 차원 확인 (채널, 높이, 너비)
print(f"Resized Image size: {image_resized.size}")  # (width, height)
print(f"Image mode: {image_resized.mode}")  # RGB인지 확인 (3채널이 맞는지 확인)

# 이미지를 텐서로 변환
transform = transforms.ToTensor()
image_tensor = transform(image_resized)

print(f"Tensor shape: {image_tensor.shape}")  # [C, H, W] 형태로 출력됨


# In[98]:


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class FireDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 이미지와 마스크 파일 경로
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # 이미지 로드
        image = Image.open(img_path).convert('RGB')  # RGB로 변환

        # 마스크 로드
        mask = Image.open(mask_path).convert('L')  # Grayscale로 변환 (마스크)

        # 변환 적용
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # 마스크를 Long Tensor로 변환 (CrossEntropyLoss를 위해)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask


# In[104]:


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

# 데이터 경로 설정
train_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/images'
train_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/train/masks'
val_image_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/images'
val_mask_dir = '/home/wuyt8807/Fast-SCNN-pytorch/data/val/masks'

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 크기 조정
    transforms.ToTensor()  # 텐서로 변환
])

# 데이터셋 로드
train_dataset = FireDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=transform)
val_dataset = FireDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=transform)

# 데이터로더 설정
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# 모델, 손실 함수, 옵티마이저 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_fast_scnn(num_classes=2).to(device)  # FastSCNN 모델 불러오기
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Mixed Precision Training
scaler = GradScaler()

# 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # 이전 그래디언트 초기화

        with autocast():  # Mixed Precision 활성화
            outputs = model(images)  # 모델의 출력
            loss = criterion(outputs[0], masks)  # 손실 계산

        # 역전파 및 옵티마이저 업데이트
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}/{len(train_loader)}: Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

    # 검증
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            with autocast():
                val_outputs = model(val_images)
                loss = criterion(val_outputs[0], val_masks)
                val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "/path/to/save/fire_fast_scnn.pth")
print("Model saved!")

