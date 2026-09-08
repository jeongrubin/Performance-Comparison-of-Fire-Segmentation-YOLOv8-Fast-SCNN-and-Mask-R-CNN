# 화재 영상 세그멘테이션 모델 비교

## 프로젝트 개요

영상에서 화재 영역을 픽셀 단위로 분할하는 세 모델(YOLOv8-Seg, Fast-SCNN, Mask R-CNN)을 비교한 프로젝트입니다. 모델별 정확도와 처리 속도를 측정하고, 원본 영상·정답 마스크·예측 마스크를 함께 확인했습니다.

화재 발생 여부만 분류하는 대신 화재가 나타난 위치와 범위를 마스크로 추정하는 것을 목표로 했습니다. CCTV나 드론처럼 영상이 연속으로 입력되는 환경을 고려해 정확도뿐 아니라 FPS도 비교했습니다.

---

## 비교한 내용

IoU, Precision, Recall, F1-score로 분할 성능을 확인하고, 같은 GPU에서 FPS를 측정했습니다. 수치만으로 확인하기 어려운 오류는 원본·정답·예측 이미지를 나란히 놓고 살펴봤습니다.

---

## 사용된 모델

서로 다른 세그멘테이션 방식을 사용하는 세 모델을 비교했습니다.

### 모델 아키텍처 비교

<img width="1202" height="678" alt="Image" src="https://github.com/user-attachments/assets/6c29e6ca-9b3c-4400-945e-f54fdb57ff92" />

| 모델         | 주요 특징                           | 아키텍처               | 장점                     |
| ---------- | ------------------------------- | ------------------ | ---------------------- |
| YOLOv8-Seg | One-Stage Detector              | Backbone-Neck-Head | 속도와 정확도의 균형            |
| Fast-SCNN  | 경량화 Semantic Segmentation       | Encoder-Decoder    | 매우 빠른 처리 속도, 저사양 환경 적합 |
| Mask R-CNN | Two-Stage Instance Segmentation | FPN + ResNet       | 정교한 객체 경계 추출, 높은 정확도   |



## 데이터셋 및 전처리

* **데이터 소스:** Roboflow - Fire Seg Part1 & Fire Segment 데이터셋 활용
* **데이터 구성:**

  * 학습(Train): 5,308장 + 954장
  * 검증(Validation): 1,213장 + 173장
  * 테스트(Test): 519장 + 56장

  보관된 Roboflow 데이터에서는 학습 이미지 수(6,262장)는 기존 기록과 일치했지만 검증·테스트 이미지 수는 달랐습니다. Fast-SCNN 재학습에는 확인된 데이터 6,262장(train), 1,558장(val), 403장(test)을 사용했습니다.

* **전처리:**

  * 모든 모델의 공정한 평가를 위해 **다각형(Polygon) 어노테이션 → 픽셀 단위 마스크(Mask)** 변환 후 사용

### 예시 이미지 
<img width="1489" height="1859" alt="Image" src="https://github.com/user-attachments/assets/af9c372f-29e5-451d-9ea8-84b415fb73cd" />
---

## 실험 결과

### 1. 정량적 성능 비교

첫 번째 표는 초기 실험에서 기록한 값입니다. 두 번째 표는 보관된 가중치와 테스트 데이터로 다시 계산한 값입니다. 재계산 값은 fire 클래스를 기준으로 한 픽셀 단위 Precision, Recall, F1, IoU이며 계산 방법은 `eval_pixel_metrics.py`에 정리했습니다.

**최초 보고 값**

| 평가지표          | YOLOv8 | Fast-SCNN | Mask R-CNN |
| ------------- | ------ | --------- | ---------- |
| **mIoU**      | 0.9810 | 0.9200    | 0.6355     |
| **Precision** | 0.9902 | 0.8010    | 0.9105     |
| **Recall**    | 0.9896 | 0.8258    | 0.6787     |
| **F1-Score**  | 0.9889 | 0.7910    | 0.7768     |

**재검증 값 (2026-09, 픽셀 단위 IoU/F1 재계산)**

| 평가지표          | YOLOv8 (n=519) | Fast-SCNN (n=403, 재학습) | Mask R-CNN (n=201) |
| ------------- | -------------- | ---------------------- | ------------------- |
| **IoU**       | 0.818          | 0.637                  | 0.643                |
| **Precision** | 0.887          | 0.690                  | 0.853                |
| **Recall**    | 0.913          | 0.892                  | 0.723                |
| **F1-Score**  | 0.900          | 0.778                  | 0.783                |

재검증 과정에서 확인한 사항:

- **Fast-SCNN**: 기존 `fast_scnn_model_fire.pth`는 현재 평가 환경에서 모든 픽셀을 배경으로 예측했습니다. 노트북에 남아 있던 학습 구조를 `models/fast_scnn_fire.py`로 정리하고 Roboflow 학습 데이터 6,262장으로 30 epoch 재학습했습니다.
- **YOLOv8-Seg**: `fire_seg_yolov8n.pt`를 테스트 이미지 519장에 적용했을 때 F1은 0.900이었습니다. 초기 기록의 Precision과 Recall(0.99대)과 차이가 있어 평가 조건을 추가로 확인할 필요가 있습니다.
- **Mask R-CNN**: 테스트 이미지 201장에서 F1 0.783, IoU 0.643으로 측정됐습니다. 초기 기록(F1 0.777, IoU 0.636)과 비슷한 값입니다.
- 모델마다 보관된 테스트셋이 달라 결과를 동일 조건의 절대 비교로 해석하기는 어렵습니다.

---

### 2. 실시간 처리 성능 (FPS)

최초 FPS는 측정 하드웨어가 명시되어 있지 않아 재현 기준으로 삼기 어렵습니다. 아래는 이 저장소 정리 시점에 **동일한 GPU(RTX 4090), 동일 영상**으로 재측정한 값입니다.

| 모델         | FPS (RTX 4090, 재측정) | 최초 보고 FPS |
| ---------- | --------------------- | ---------- |
| Fast-SCNN  | 706.7                 | 164.13     |
| YOLOv8     | 112.7                 | 91.30      |
| Mask R-CNN | 29.9                  | 3.29       |

절대값은 하드웨어 차이로 다르지만, **Fast-SCNN > YOLOv8 > Mask R-CNN 순의 상대적 속도 순위는 재측정에서도 동일하게 유지**됩니다.

### 3. 실제 영상에서의 동작 (GIF)

프레임별 예측 변화를 확인하기 위해 모델별 추론 결과를 GIF로 만들었습니다(`build_gifs.py`). YOLOv8-Seg와 Fast-SCNN은 `화재2.mp4`를 사용했고, Mask R-CNN은 장갑차 화재 영상을 사용했습니다. 입력 영상이 서로 다르므로 이 자료는 정성적 동작 확인용입니다.

| YOLOv8-Seg | Fast-SCNN (재학습) | Mask R-CNN |
|---|---|---|
| ![YOLO demo](assets/demo/yolo_demo.gif) | ![Fast-SCNN demo](assets/demo/fastscnn_demo.gif) | ![Mask R-CNN demo](assets/demo/maskrcnn_demo.gif) |

---

## 테스트 이미지 모델별 Ground Truth vs Prediction 비교

예측 결과를 확인할 수 있도록 각 모델의 테스트 이미지에서 원본, 정답 마스크, 예측 마스크를 나란히 배치했습니다. 이미지는 `build_gt_comparison.py`로 생성했습니다.

### YOLOv8-Seg
![YOLOv8-Seg: Original / GT / Prediction](assets/predictions/yolo_predicted.png)

### Mask R-CNN
정답에는 화재 영역이 두 곳이지만 예측은 한 곳만 검출한 사례가 보입니다. 누락이 발생하는 경향은 Recall 0.723에서도 확인됩니다.

![Mask R-CNN: Original / GT / Prediction](assets/predictions/mask_rcnn_predicted.png)

### Fast-SCNN (재학습)
예측 경계가 비교적 거칠고 배경 일부를 화재로 분류하는 사례가 나타났습니다. Precision은 0.690으로 측정됐습니다.

![Fast-SCNN: Original / GT / Prediction](assets/predictions/fast_scnn_predicted.png)

---

## 결과에서 확인한 점

재검증 결과에서는 YOLOv8-Seg의 F1과 IoU가 가장 높았고, Fast-SCNN의 처리 속도가 가장 빨랐습니다. Mask R-CNN은 초기 기록과 다시 측정한 값의 차이가 가장 작았습니다. 모델마다 보관된 테스트셋이 달라 세 수치를 완전히 같은 조건의 순위로 해석하지는 않았습니다.

### 프로젝트 최종 개념도

<img width="914" height="737" alt="Image" src="https://github.com/user-attachments/assets/d3bb8039-b0ab-4e61-a617-2434090683e2" />

---

## 주요 파일

- `train_fastscnn_fire.py`: Fast-SCNN 학습
- `eval_pixel_metrics.py`: 모델별 픽셀 단위 지표 계산
- `build_gt_comparison.py`: 원본·정답·예측 비교 이미지 생성
- `build_gifs.py`: 영상 추론 결과 GIF 생성
- `models/fast_scnn_fire.py`: 재학습에 사용한 Fast-SCNN 구조

YOLOv8-Seg는 Ultralytics CLI(`yolo segment train ...`)로 별도 학습했습니다.

> **참고**: `train_eval_fast_scnn.py`는 노트북을 그대로 스크립트로 변환한 것이라 그 안에 여러 버전의 아키텍처 실험이 섞여 있습니다. 실제 저장된 화재 탐지 체크포인트와 정확히 일치하는 것은 `models/fast_scnn_fire.py` 하나뿐이며, 이 사실은 체크포인트의 state_dict 키를 직접 대조해 확인했습니다.

## 실행 방법

원본 화재 CCTV/차량 영상과 대용량 학습 데이터는 포함하지 않았습니다. 아래는 재현을 위한 최소 절차입니다.

```bash
pip install -r requirements.txt

# Fast-SCNN 학습 (data/{train,val,test}/{images,masks} 구조 필요)
python train_fastscnn_fire.py

# 세 모델 공통 픽셀 단위 성능 평가
python eval_pixel_metrics.py --model fast_scnn --weights fast_scnn_fire.pth \
    --images data/test/images --labels data/test/masks

# YOLOv8-Seg (Ultralytics)
yolo segment train data=fire_seg.yaml model=yolov8n-seg.pt
```

`data_loader/fire_dataset.py`는 Roboflow에서 내려받은 `Images/`, `Masks/` 폴더 구조를 기준으로 작성되어 있어, 동일한 디렉터리 구조로 데이터를 배치하면 그대로 사용할 수 있습니다.

---

## 남은 문제

초기 YOLOv8 기록과 재검증 값의 차이가 발생한 원인은 아직 특정하지 못했습니다. 다음 실험에서는 데이터 분할을 영상 단위로 고정하고, 세 모델을 같은 테스트셋에서 다시 평가할 계획입니다. 야간·연기 환경의 데이터도 추가로 필요합니다.

---

## 참고 문헌

* Ultralytics YOLOv8
* Poudel, R. P. K., et al. *Fast-SCNN: Fast Semantic Segmentation Network*
* He, K., et al. *Mask R-CNN*
* Detectron2
