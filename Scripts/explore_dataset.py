import os
import matplotlib.pyplot as plt

def explore_local_dataset():
    """Explore the local dataset structure"""
    dataset_path = "./dataset"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        print("Please create a 'dataset' folder and add your images")
        return False
    
    print("📁 Exploring dataset structure...")
    print("=" * 50)
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not classes:
        print("❌ No class folders found in dataset directory")
        return False
    
    print(f"Found {len(classes)} classes:")
    
    class_counts = {}
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        
        # Count image files
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        
        image_count = len(image_files)
        class_counts[class_name] = image_count
        total_images += image_count
        
        print(f"  📂 {class_name}: {image_count} images")
        
        # Show first few image names
        if image_files:
            print(f"     Sample files: {', '.join(image_files[:3])}")
            if len(image_files) > 3:
                print(f"     ... and {len(image_files) - 3} more")
        print()
    
    print(f"📊 Total images: {total_images}")
    
    # Check for Newcastle disease
    newcastle_classes = [cls for cls in classes if 'newcastle' in cls.lower()]
    if newcastle_classes:
        print(f"✅ Newcastle disease classes found: {newcastle_classes}")
    else:
        print(f"⚠️ No Newcastle disease class found")
        print(f"Available classes: {classes}")
    
    # Plot class distribution
    if class_counts:
        plt.figure(figsize=(12, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    return True

if __name__ == "__main__":
    success = explore_local_dataset()
    if success:
        print("🎉 Dataset is ready for training!")
        print("You can now run: python train_models.py")
    else:
        print("💡 Please organize your dataset and try again")