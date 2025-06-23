import kagglehub
import os
import shutil

def download_and_setup_dataset():
    """Download and setup the poultry disease dataset"""
    print("Starting dataset download...")
    
    try:
        # Download the dataset
        print("Downloading dataset from Kaggle...")
        path = kagglehub.dataset_download("chandrashekarnatesh/poultry-diseases")
        print(f"Dataset downloaded to: {path}")
        
        # Create local dataset directory
        local_dataset_path = "./dataset"
        
        # Remove existing dataset if it exists
        if os.path.exists(local_dataset_path):
            print("Removing existing dataset directory...")
            shutil.rmtree(local_dataset_path)
        
        # Copy dataset to local directory
        print("Copying dataset to local directory...")
        shutil.copytree(path, local_dataset_path)
        print(f"Dataset copied to: {local_dataset_path}")
        
        # Explore dataset structure
        print("\n" + "="*50)
        print("DATASET STRUCTURE")
        print("="*50)
        
        total_images = 0
        class_info = {}
        
        for root, dirs, files in os.walk(local_dataset_path):
            level = root.replace(local_dataset_path, '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            
            if level == 0:
                print(f"{indent}📁 {folder_name}/")
            elif level == 1:
                # Count images in this class
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
                image_count = len(image_files)
                total_images += image_count
                class_info[folder_name] = image_count
                
                print(f"{indent}📂 {folder_name}/ ({image_count} images)")
                
                # Show first few image names
                subindent = ' ' * 2 * (level + 1)
                for i, file in enumerate(image_files[:3]):
                    print(f"{subindent}🖼️ {file}")
                if len(image_files) > 3:
                    print(f"{subindent}... and {len(image_files) - 3} more images")
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"Total classes: {len(class_info)}")
        print(f"Total images: {total_images}")
        
        print(f"\n📋 CLASS BREAKDOWN:")
        for class_name, count in class_info.items():
            percentage = (count / total_images) * 100
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        # Check for Newcastle disease
        newcastle_classes = [cls for cls in class_info.keys() if 'newcastle' in cls.lower()]
        if newcastle_classes:
            print(f"\n✅ Newcastle disease classes found: {newcastle_classes}")
        else:
            print(f"\n⚠️ No Newcastle disease class found. Available classes: {list(class_info.keys())}")
        
        print(f"\n🎉 Dataset setup completed successfully!")
        print(f"You can now run: python train_models.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("Please check your internet connection and try again.")
        return False

if __name__ == "__main__":
    success = download_and_setup_dataset()
    if not success:
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure you have internet connection")
        print("2. Try running: pip install --upgrade kagglehub")
        print("3. Check if you have enough disk space")