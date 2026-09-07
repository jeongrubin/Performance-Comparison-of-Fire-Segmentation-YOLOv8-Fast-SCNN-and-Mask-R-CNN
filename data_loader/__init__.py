from .fire_dataset import FireDataset  # FireDataset 클래스 import

datasets = {
    'fire': FireDataset,  # 화재 데이터셋만 등록
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Dataset"""
    if name.lower() not in datasets:
        raise ValueError(f"Dataset {name} is not registered! Available datasets: {list(datasets.keys())}")
    return datasets[name.lower()](**kwargs)
