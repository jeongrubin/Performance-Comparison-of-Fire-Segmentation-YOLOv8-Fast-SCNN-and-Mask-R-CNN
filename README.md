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

##  사용된 모델

본 연구에서는 각기 다른 접근 방식을 대표하는 세 가지 모델을 선정하여 다각적인 비교를 수행함

### 모델 아키텍처 비교

<img width="1202" height="678" alt="Image" src="https://github.com/user-attachments/assets/6c29e6ca-9b3c-4400-945e-f54fdb57ff92" />

| 모델         | 주요 특징                           | 아키텍처               | 장점                     |
| ---------- | ------------------------------- | ------------------ | ---------------------- |
| YOLOv8-Seg | One-Stage Detector              | Backbone-Neck-Head | 속도와 정확도의 균형            |
| Fast-SCNN  | 경량화 Semantic Segmentation       | Encoder-Decoder    | 매우 빠른 처리 속도, 저사양 환경 적합 |
| Mask R-CNN | Two-Stage Instance Segmentation | FPN + ResNet       | 정교한 객체 경계 추출, 높은 정확도   |

---

## 데이터셋 및 전처리

* **데이터 소스:** Roboflow - Fire Seg Part1 & Fire Segment 데이터셋 활용
* **데이터 구성:**

  * 학습(Train): 5,308장 + 954장
  * 검증(Validation): 1,213장 + 173장
  * 테스트(Test): 519장 + 56장
    
* **전처리:**

  * 모든 모델의 공정한 평가를 위해 **다각형(Polygon) 어노테이션 → 픽셀 단위 마스크(Mask)** 변환 후 사용


---

## 실험 결과

### 1. 정량적 성능 비교

각 모델을 50 에포크(Epoch) 동안 학습시킨 후, 테스트 데이터셋으로 성능을 측정함

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

# 테스트 이미지 모델별 Mask 예측 

### YOLO 
<img width="1490" height="1107" alt="Image" src="https://github.com/user-attachments/assets/101cfbd3-aadd-4c37-a778-e1c21f100a6c" />

### Mask R-CNN
<img width="1490" height="1107" alt="Image" src="https://github.com/user-attachments/assets/d8c88dc0-178c-423c-99ed-afa4a7a1d83b" />

##Fast RCNN
<img width="3150" height="3150" alt="Image" src="https://github.com/user-attachments/assets/26a62693-f5d3-4680-980d-16c166095b84" />


### 모델별 동영상 예측 결과 비교

<img width="881" height="307" alt="Image" src="https://github.com/user-attachments/assets/51692025-aec1-42e0-a37d-13be6981cef3" />

---

#

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
