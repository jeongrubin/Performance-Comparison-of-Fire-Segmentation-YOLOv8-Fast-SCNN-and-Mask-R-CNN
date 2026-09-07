class FireDataset(Dataset):
    """Fire Segmentation Dataset."""

    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Define paths for images and masks
        self.image_dir = os.path.join(root, split + '_images')  # 예: 'train_images' 또는 'val_images'
        self.mask_dir = os.path.join(root, split + '_masks')    # 예: 'train_masks' 또는 'val_masks'

        # Check if paths exist
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Get sorted lists of images and masks
        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        # Ensure that the images and masks match
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"
        for img, mask in zip(self.image_files, self.mask_files):
            assert os.path.splitext(img)[0] == os.path.splitext(mask)[0], \
                f"Image {img} and mask {mask} do not match"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[index])
        mask_path = os.path.join(self.mask_dir, self.mask_files[index])

        # Load as numpy arrays
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask at {mask_path}")

        # Convert numpy arrays to PIL Images
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        # Apply transformations if specified
        if self.transform:
            image, mask = self.transform(image, mask)  # Apply transformations

        # Convert mask to tensor (1 채널로 유지)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
