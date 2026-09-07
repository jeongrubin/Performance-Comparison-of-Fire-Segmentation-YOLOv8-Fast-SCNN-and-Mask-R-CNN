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

  실제 로컬에 남아있던 Roboflow 스냅샷 기준으로는 검증/테스트 장수가 위 수치와 정확히 일치하지 않았습니다(학습 장수 6,262장은 정확히 일치). Fast-SCNN 재학습에는 확보 가능한 전체 6,262장(train) / 1,558장(val) / 403장(test)을 사용했습니다.

* **전처리:**

  * 모든 모델의 공정한 평가를 위해 **다각형(Polygon) 어노테이션 → 픽셀 단위 마스크(Mask)** 변환 후 사용

### 예시 이미지 
<img width="1489" height="1859" alt="Image" src="https://github.com/user-attachments/assets/af9c372f-29e5-451d-9ea8-84b415fb73cd" />
---

## 실험 결과

### 1. 정량적 성능 비교

아래 두 표는 서로 다른 시점의 수치입니다. 위쪽은 논문/최초 실험 당시 보고한 값이고, 아래쪽은 이 저장소를 정리하면서 **각 모델의 실제 가중치 파일로 보유 중인 held-out 테스트셋에 대해 직접 재실행**하여 얻은 값입니다(픽셀 단위 Precision/Recall/F1/IoU, fire 클래스 기준. 계산 코드는 `eval_pixel_metrics.py` 참고).

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

재검증 과정에서 확인된 사실:

- **Fast-SCNN**: 저장되어 있던 `fast_scnn_model_fire.pth`는 자체 검증 데이터에서조차 fire 클래스를 전혀 예측하지 못하는(모든 픽셀을 배경으로 예측) 손상/미완성 체크포인트였습니다. 노트북에 남아있던 실제 학습 코드와 아키텍처(`models/fast_scnn_fire.py`)를 복원해 Roboflow 원본 데이터(6,262장)로 30 epoch 재학습했고, 위 수치는 그 결과입니다.
- **YOLOv8-Seg**: 가중치(`fire_seg_yolov8n.pt`) 자체는 정상 동작하지만, 원래 보고된 Precision/Recall(0.99대)은 매칭되는 테스트셋(519장, Ultralytics 공식 `val()` 및 픽셀 단위 재계산 모두)으로 재현되지 않았습니다. 실제 재현되는 값은 F1 0.90 수준입니다.
- **Mask R-CNN**: 가중치(`mask_rcnn_model.pth`)와 최초 보고 수치가 held-out 테스트셋(201장)에서 **가장 근접하게 재현**되었습니다(F1 0.777 → 0.783, IoU 0.636 → 0.643).
- 세 모델은 각각 다른 데이터 파이프라인으로 학습되어 테스트셋 구성이 완전히 동일하지 않습니다(모델별로 보유하고 있던 held-out 세트를 그대로 사용).

---

### 2. 실시간 처리 성능 (FPS)

최초 FPS는 측정 하드웨어가 명시되어 있지 않아 재현 기준으로 삼기 어렵습니다. 아래는 이 저장소 정리 시점에 **동일한 GPU(RTX 4090), 동일 영상**으로 재측정한 값입니다.

| 모델         | FPS (RTX 4090, 재측정) | 최초 보고 FPS |
| ---------- | --------------------- | ---------- |
| Fast-SCNN  | 706.7                 | 164.13     |
| YOLOv8     | 112.7                 | 91.30      |
| Mask R-CNN | 29.9                  | 3.29       |

절대값은 하드웨어 차이로 다르지만, **Fast-SCNN > YOLOv8 > Mask R-CNN 순의 상대적 속도 순위는 재측정에서도 동일하게 유지**됩니다.

## 테스트 이미지 모델별 Ground Truth vs Prediction 비교

정성적 결과를 예측 마스크만 보여주면 "이게 맞게 예측한 건지" 판단할 근거가 없어, 각 모델의 held-out 테스트셋에서 무작위로 뽑은 이미지에 대해 **원본 / 정답(Ground Truth) / 예측(Prediction)을 나란히** 배치했습니다(모두 이 저장소 정리 시점에 직접 재실행한 결과, 생성 코드는 `build_gt_comparison.py`).

### YOLOv8-Seg
![YOLOv8-Seg: Original / GT / Prediction](assets/predictions/yolo_predicted.png)

### Mask R-CNN
GT에 화재 영역이 2곳인데 예측은 1곳만 잡는 경우가 반복적으로 보입니다 — 재검증한 Recall 0.723과 일치하는 정직한 결과입니다.

![Mask R-CNN: Original / GT / Prediction](assets/predictions/mask_rcnn_predicted.png)

### Fast-SCNN (재학습)
가장 가벼운 모델답게 경계가 거칠고 일부 영역을 놓치는 경향이 있습니다 — 세 모델 중 가장 낮은 Precision(0.690)과 일치합니다.

![Fast-SCNN: Original / GT / Prediction](assets/predictions/fast_scnn_predicted.png)

---

## 결론 및 분석

재검증 수치(픽셀 단위 F1/IoU, FPS는 RTX 4090 재측정) 기준으로 정리하면:

* **YOLOv8: The Best All-Rounder**

  * F1 0.90, IoU 0.82로 세 모델 중 가장 균형적인 정확도. FPS도 세 모델 중 두 번째로 빨라(112.7) 정확도와 실시간성을 모두 요구하는 환경에 적합.

* **Fast-SCNN: The Speed Champion**

  * 706.7 FPS로 압도적으로 빠름(YOLO 대비 약 6배) → 드론, 임베디드 등 연산 자원이 제한적인 환경에 적합. 다만 정확도(F1 0.78)는 세 모델 중 가장 낮아, 속도와 정확도의 트레이드오프가 뚜렷함.

* **Mask R-CNN: 가장 무거운, 그러나 가장 재현성 높은 모델**

  * FPS 29.9로 세 모델 중 가장 느려 실시간 탐지에는 부적합. 정확도(F1 0.78)는 Fast-SCNN과 비슷한 수준이었고, 세 모델 중 최초 보고 수치와 실측치가 가장 근접해 재현성이 가장 높았음.

### 프로젝트 최종 개념도

<img width="914" height="737" alt="Image" src="https://github.com/user-attachments/assets/d3bb8039-b0ab-4e61-a617-2434090683e2" />

---

## 코드 구성

```text
.
├── models/
│   ├── fast_scnn_fire.py       # 실제 화재 체크포인트와 일치하는 아키텍처 (검증됨, fast_scnn_fire.pth 로드용)
│   └── fast_scnn.py            # 원본 Fast-SCNN(Poudel et al.) 구현체 — upstream 참고용, 화재 체크포인트와는 무관
├── data_loader/                 # fire_dataset.py — Roboflow 화재 세그멘테이션 데이터셋 로더
├── utils/                       # loss, lr_scheduler, metric(mIoU), visualize
├── train.py / eval.py / demo.py    # 원본 Fast-SCNN(Poudel et al.) 학습/평가/추론 스캐폴드
├── train_fastscnn_fire.py       # 실제 사용하는 학습 스크립트 — models/fast_scnn_fire.py 기반, fast_scnn_fire.pth 생성
├── fast_scnn_fire.pth           # 위 스크립트로 학습된 체크포인트 (299KB, 재현 가능)
├── eval_pixel_metrics.py        # 세 모델 공통 픽셀 단위 Precision/Recall/F1/IoU 평가 코드
├── build_gt_comparison.py       # Original/GT/Prediction 비교 이미지 생성 코드
├── train_eval_fast_scnn.py      # Fast-SCNN 실험 노트북 원본(정리본) — 여러 아키텍처 실험 이력 포함
├── train_eval_mask_rcnn.py      # torchvision Mask R-CNN(ResNet50-FPN) 학습 및 평가 노트북(정리본)
└── experiment_deeplabv3.py      # DeepLabv3 비교 실험(참고용, 최종 결과표에는 미포함)
```

YOLOv8-Seg는 Ultralytics CLI(`yolo segment train ...`)로 별도 학습했습니다.

> **참고**: `train_eval_fast_scnn.py`는 노트북을 그대로 스크립트로 변환한 것이라 그 안에 여러 버전의 아키텍처 실험이 섞여 있습니다. 실제 저장된 화재 탐지 체크포인트와 정확히 일치하는 것은 `models/fast_scnn_fire.py` 하나뿐이며, 이 사실은 체크포인트의 state_dict 키를 직접 대조해 확인했습니다.

## 실행 방법 (참고용)

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

## 향후 개선 (Future Work)

* **Fast-SCNN 성능 개선:** 현재 픽셀 F1 0.78 수준으로 세 모델 중 가장 낮음. 지식 증류(Knowledge Distillation) 등 최신 기법 적용.
* **YOLOv8 재검증 격차 원인 분석:** 최초 보고 수치(F1 0.99)와 재검증 수치(F1 0.90) 간 격차의 원인(평가 스크립트 차이, confidence threshold 등)을 특정하지 못함.
* **데이터 분할 방식 점검:** Mask R-CNN 검증셋 일부가 연속 프레임으로 구성되어 있어, 학습/검증 분할이 프레임 단위가 아닌 영상(클립) 단위로 이루어졌는지 확인이 필요함.
* **데이터 다양성 확보:** 야간, 실내, 연기 환경 등 다양한 데이터 확보.
* **Transformer 기반 모델 탐색:** 최신 Vision Transformer 기반 세그멘테이션과 비교.

---

## 참고 문헌

* Ultralytics YOLOv8
* Poudel, R. P. K., et al. *Fast-SCNN: Fast Semantic Segmentation Network*
* He, K., et al. *Mask R-CNN*
* Detectron2
