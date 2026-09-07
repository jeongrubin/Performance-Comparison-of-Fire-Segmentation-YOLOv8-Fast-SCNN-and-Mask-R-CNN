# Performance Comparison of Fire Segmentation: YOLOv8,Fast-SCNN, and Mask R-CNN

## 프로젝트 개요

본 프로젝트는 실시간 영상에서 화재 영역을 정밀하게 탐지하기 위해 대표적인 딥러닝 세그멘테이션 모델인 **YOLOv8, Fast-SCNN, Mask R-CNN**의 성능을 비교 분석합니다.

기존의 온도, 연기 센서 기반 화재 감지 시스템은 화재가 일정 수준 확산된 후에야 작동하는 한계가 있습니다. 저희는 딥러닝 영상 분석 기술을 통해 이러한 문제를 해결하고자 합니다. 단순 화재 *감지*를 넘어, 화재 영역을 픽셀 단위로 정확하게 *분할(Segmentation)* 함으로써 화재의 확산 범위, 형태 등을 초기 단계부터 정밀하게 파악하는 것을 목표로 합니다.

궁극적으로는 **소방 드론, 지능형 CCTV, 재난 대응 로봇** 등 차세대 안전 시스템에 적용 가능한 최적의 딥러닝 모델을 탐색하고 그 실증적 근거를 제시하고자 합니다.

---

## 주요 목표

* **정확성 vs. 속도 트레이드오프 분석:** 각 모델의 mIoU, F1-Score와 FPS를 비교하여 사용 목적에 맞는 최적의 모델 탐색
* **실시간 적용 가능성 검증:** 실제 동영상 데이터를 활용하여 동적 환경에서의 모델별 성능을 시각적으로 비교 및 분석
* **모델별 강점 및 약점 도출:** 실시간 감시, 정밀 분석 등 특정 시나리오에 어떤 모델이 더 적합한지에 대한 명확한 기준 제시

---

## 사용된 모델

본 연구에서는 각기 다른 접근 방식을 대표하는 세 가지 모델을 선정하여 다각적인 비교를 수행함

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
    
* **전처리:**

  * 모든 모델의 공정한 평가를 위해 **다각형(Polygon) 어노테이션 → 픽셀 단위 마스크(Mask)** 변환 후 사용

### 예시 이미지 
<img width="1489" height="1859" alt="Image" src="https://github.com/user-attachments/assets/af9c372f-29e5-451d-9ea8-84b415fb73cd" />
---

## 실험 결과

### 1. 정량적 성능 비교

각 모델을 학습시킨 후, 테스트 데이터셋으로 성능을 측정함

| 평가지표          | YOLOv8 | Fast-SCNN | Mask R-CNN |
| ------------- | ------ | --------- | ---------- |
| **mIoU**      | 0.9810 | 0.9200    | 0.6355     |
| **Precision** | 0.9902 | 0.8010    | 0.9105     |
| **Recall**    | 0.9896 | 0.8258    | 0.6787     |
| **F1-Score**  | 0.9889 | 0.7910    | 0.7768     |

---

### 2. 실시간 처리 성능 (FPS)

차량 화재 동영상 데이터를 이용해 초당 프레임 처리 속도(FPS)를 측정함

| 모델         | FPS    |
| ---------- | ------ |
| Fast-SCNN  | 164.13 |
| YOLOv8     | 91.30  |
| Mask R-CNN | 3.29   |

## 테스트 이미지 모델별 Mask 예측

각 모델을 서로 다른 화재 영상(차량 화재 / 장갑차 화재)에 적용한 프레임 예시입니다.

### YOLO
![YOLOv8-Seg 예측 결과](assets/predictions/yolo_predicted.png)

### Mask R-CNN
![Mask R-CNN 예측 결과](assets/predictions/mask_rcnn_predicted.png)

### Fast-SCNN
![Fast-SCNN 예측 결과](assets/predictions/fast_scnn_predicted.png)


---

## 결론 및 분석

* **YOLOv8: The Best All-Rounder**

  * mIoU와 F1-Score 최고, FPS도 준수 → 정확도와 실시간성을 모두 요구하는 환경에 적합.

* **Fast-SCNN: The Speed Champion**

  * 164 FPS의 압도적 속도 → 드론, 로봇 등 임베디드 환경에서 유용.

* **Mask R-CNN: The Precision Specialist**

  * 정교한 경계 추출 가능, 하지만 FPS 3.29로 실시간 탐지에는 부적합 → 정적 이미지 정밀 분석에 적합.

### 프로젝트 최종 개념도

<img width="914" height="737" alt="Image" src="https://github.com/user-attachments/assets/d3bb8039-b0ab-4e61-a617-2434090683e2" />

---

## 코드 구성

```text
.
├── models/                    # fast_scnn.py, fast_scnn2.py
├── data_loader/                # fire_dataset.py — Roboflow 화재 세그멘테이션 데이터셋 로더
├── utils/                      # loss, lr_scheduler, metric(mIoU), visualize
├── train.py / eval.py / demo.py   # Fast-SCNN 학습/평가/추론 (Poudel et al. 구현체 기반, 화재 데이터셋용으로 수정)
├── train_eval_fast_scnn.py     # Fast-SCNN 학습 및 평가 노트북(정리본)
├── train_eval_mask_rcnn.py     # torchvision Mask R-CNN(ResNet50-FPN) 학습 및 평가 노트북(정리본)
└── experiment_deeplabv3.py     # DeepLabv3 비교 실험(참고용, 최종 결과표에는 미포함)
```

YOLOv8-Seg는 Ultralytics CLI(`yolo segment train ...`)로 별도 학습했으며, 위 표의 성능/FPS 수치에 사용된 가중치입니다.

## 실행 방법 (참고용)

원본 화재 CCTV/차량 영상과 학습 가중치는 용량 및 출처 문제로 포함하지 않았습니다. 아래는 재현을 위한 최소 절차입니다.

```bash
pip install -r requirements.txt

# Fast-SCNN 학습/평가
python train_eval_fast_scnn.py

# Mask R-CNN 학습/평가
python train_eval_mask_rcnn.py

# YOLOv8-Seg (Ultralytics)
yolo segment train data=fire_seg.yaml model=yolov8n-seg.pt
```

`data_loader/fire_dataset.py`는 Roboflow에서 내려받은 `Images/`, `Masks/` 폴더 구조를 기준으로 작성되어 있어, 동일한 디렉터리 구조로 데이터를 배치하면 그대로 사용할 수 있습니다.

---

## 향후 개선 (Future Work)

* **Fast-SCNN 성능 개선:** 지식 증류(Knowledge Distillation) 등 최신 기법 적용.
* **데이터 다양성 확보:** 야간, 실내, 연기 환경 등 다양한 데이터 확보.
* **Transformer 기반 모델 탐색:** 최신 Vision Transformer 기반 세그멘테이션과 비교.

---

## 참고 문헌

* Ultralytics YOLOv8
* Poudel, R. P. K., et al. *Fast-SCNN: Fast Semantic Segmentation Network*
* He, K., et al. *Mask R-CNN*
* Detectron2
