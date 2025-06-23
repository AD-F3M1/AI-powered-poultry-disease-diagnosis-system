# Optimized Binary Classification for 8GB RAM systems
import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from datetime import datetime
import psutil

# OPTIMIZED CONFIGURATION for 8GB RAM + Intel UHD 620
DATASET_PATH = r"C:\Users\Admin\Desktop\AI_Poultry_Disease_Diagnosis\dataset"
IMAGE_SIZE = 128  # Reduced from 224 to save memory
BATCH_SIZE = 16   # Reduced from 64 to prevent OOM
EPOCHS = 15       # Slightly increased since each epoch is faster
MODEL_SAVE_PATH = r"C:\Users\Admin\Desktop\AI_Poultry_Disease_Diagnosis\models"

# Memory management settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
tf.config.threading.set_intra_op_parallelism_threads(4)  # Match your CPU cores
tf.config.threading.set_inter_op_parallelism_threads(2)

# Configure memory growth for integrated GPU
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(f"GPU configuration: {e}")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def log_message(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    memory_usage = psutil.virtual_memory().percent
    print(f"[{timestamp}] [RAM: {memory_usage:.1f}%] {message}")

def monitor_memory():
    """Monitor system memory usage"""
    memory = psutil.virtual_memory()
    if memory.percent > 85:
        log_message("⚠️ HIGH MEMORY USAGE - Running garbage collection")
        gc.collect()
        tf.keras.backend.clear_session()

def check_binary_dataset():
    """Check if we have exactly 2 classes"""
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder not found!")
        print(f"Expected path: {DATASET_PATH}")
        return False
    
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    if len(classes) != 2:
        print(f"❌ Expected 2 classes, found {len(classes)}: {classes}")
        return False
    
    log_message(f"✅ Binary classification setup detected:")
    
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        image_count = len(images)
        total_images += image_count
        log_message(f"  📂 {class_name}: {image_count:,} images")
    
    log_message(f"📊 Total images: {total_images:,}")
    
    # Memory estimation
    estimated_memory = (total_images * IMAGE_SIZE * IMAGE_SIZE * 3 * 4) / (1024**3)  # 4 bytes per float32
    log_message(f"💾 Estimated memory needed: {estimated_memory:.2f} GB")
    
    if estimated_memory > 6:  # Leave 2GB for system
        log_message("⚠️ Dataset might be large for 8GB RAM - using smaller batches")
        global BATCH_SIZE
        BATCH_SIZE = max(8, BATCH_SIZE // 2)
    
    return True

def create_optimized_generators():
    """Create memory-efficient data generators"""
    log_message("Creating optimized data generators...")
    
    # Lighter data augmentation to save processing time
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,      # Reduced from 15
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,       # Reduced from 0.1
        zoom_range=0.05,        # Reduced from 0.1
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )
    
    log_message(f"Training samples: {train_generator.samples:,}")
    log_message(f"Validation samples: {validation_generator.samples:,}")
    log_message(f"Batch size: {BATCH_SIZE}")
    log_message(f"Steps per epoch: {train_generator.samples // BATCH_SIZE}")
    
    return train_generator, validation_generator

def build_lightweight_cnn():
    """Build a lighter CNN optimized for 8GB RAM"""
    log_message("Building lightweight CNN for 8GB RAM...")
    
    model = Sequential([
        # First block - reduced filters
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second block
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers - reduced size
        Flatten(),
        Dense(128, activation='relu'),  # Reduced from 512
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Use a lower learning rate for stability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Reduced from 0.001
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_with_memory_management(model, train_gen, val_gen):
    """Train with memory monitoring and management"""
    log_message("Starting memory-optimized training...")
    
    # Custom callback for memory monitoring
    class MemoryCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            monitor_memory()
            memory_percent = psutil.virtual_memory().percent
            log_message(f"Epoch {epoch + 1} starting - RAM usage: {memory_percent:.1f}%")
        
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()  # Force garbage collection after each epoch
    
    callbacks = [
        MemoryCallback(),
        EarlyStopping(
            monitor='val_loss', 
            patience=7,  # Increased patience for slower convergence
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=4,  # Reduced patience
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, 'best_lightweight_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1,
            save_weights_only=False
        )
    ]
    
    # Train with progress tracking
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1,
        workers=2,  # Limit workers to save memory
        use_multiprocessing=False  # Disable multiprocessing for stability
    )
    
    return history

def plot_optimized_results(history):
    """Plot training results with memory-efficient plotting"""
    plt.style.use('default')  # Use default style to save memory
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy plot
    axes[0, 0].plot(history.history['accuracy'], label='Training', linewidth=2, color='blue')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, color='orange')
    axes[0, 0].set_title('Model Accuracy', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(history.history['loss'], label='Training', linewidth=2, color='blue')
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2, color='orange')
    axes[0, 1].set_title('Model Loss', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training progress
    epochs = len(history.history['accuracy'])
    axes[1, 0].bar(['Epochs Completed'], [epochs], color='green', alpha=0.7)
    axes[1, 0].set_title('Training Progress')
    axes[1, 0].set_ylabel('Epochs')
    
    # Final metrics
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    
    metrics = ['Val Accuracy', 'Val Loss']
    values = [final_acc, final_loss]
    colors = ['green', 'red']
    
    bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Final Metrics')
    axes[1, 1].set_ylabel('Value')
    
    # Add value labels
    for bar, value in zip(bars, values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'lightweight_training_results.png'), 
                dpi=150, bbox_inches='tight')  # Reduced DPI to save memory
    plt.show()
    plt.close()  # Close to free memory

def evaluate_lightweight_model(model, val_gen, class_indices):
    """Memory-efficient model evaluation"""
    log_message("Evaluating lightweight model...")
    monitor_memory()
    
    # Get predictions in smaller batches to save memory
    val_gen.reset()
    predictions = []
    y_true = []
    
    steps = val_gen.samples // val_gen.batch_size
    for i in range(steps):
        batch_x, batch_y = next(val_gen)
        batch_pred = model.predict(batch_x, verbose=0)
        predictions.extend(batch_pred.flatten())
        y_true.extend(batch_y)
        
        if i % 10 == 0:  # Monitor every 10 batches
            monitor_memory()
    
    y_pred_prob = np.array(predictions)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = np.array(y_true)
    
    # Get class names
    class_names = list(class_indices.keys())
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    print("\n" + "="*60)
    print("🎯 LIGHTWEIGHT MODEL RESULTS")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Create evaluation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'lightweight_evaluation.png'), 
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    
    # Newcastle Disease specific metrics
    newcastle_idx = None
    for idx, name in class_indices.items():
        if 'castle' in name.lower() or 'newcastle' in name.lower():
            newcastle_idx = idx
            break
    
    if newcastle_idx is not None:
        newcastle_name = class_names[newcastle_idx]
        precision = report[newcastle_name]['precision']
        recall = report[newcastle_name]['recall']
        f1 = report[newcastle_name]['f1-score']
        
        print("\n" + "🦠" * 20)
        print("NEWCASTLE DISEASE DETECTION")
        print("🦠" * 20)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"AUC-ROC:   {roc_auc:.4f}")
        
        # Performance assessment
        if precision >= 0.90 and recall >= 0.90:
            print("🎉 EXCELLENT performance for 8GB RAM system!")
        elif precision >= 0.80 and recall >= 0.80:
            print("✅ GOOD performance - suitable for practical use")
        elif precision >= 0.70 and recall >= 0.70:
            print("👍 ACCEPTABLE performance - consider more training")
        else:
            print("⚠️ NEEDS IMPROVEMENT - try different parameters")
    
    return report['accuracy'], roc_auc

def main():
    """Main optimized training function"""
    print("\n" + "🐔" * 25)
    print("OPTIMIZED AI POULTRY DISEASE DIAGNOSIS")
    print("FOR 8GB RAM + Intel UHD 620")
    print("🐔" * 25)
    
    # System info
    memory = psutil.virtual_memory()
    log_message(f"System RAM: {memory.total / (1024**3):.1f} GB")
    log_message(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    
    # Check dataset
    if not check_binary_dataset():
        print("Please fix your dataset structure and try again.")
        return
    
    # Create optimized data generators
    train_gen, val_gen = create_optimized_generators()
    class_indices = train_gen.class_indices
    
    log_message(f"Class mapping: {class_indices}")
    
    # Build lightweight model
    model = build_lightweight_cnn()
    
    print("\n" + "="*50)
    print("LIGHTWEIGHT MODEL ARCHITECTURE")
    print("="*50)
    model.summary()
    
    # Calculate model size
    param_count = model.count_params()
    model_size_mb = param_count * 4 / (1024**2)  # 4 bytes per parameter
    log_message(f"Model size: ~{model_size_mb:.1f} MB")
    
    # Train model
    print("\n" + "="*50)
    print("STARTING OPTIMIZED TRAINING")
    print("="*50)
    
    history = train_with_memory_management(model, train_gen, val_gen)
    
    # Plot results
    plot_optimized_results(history)
    
    # Evaluate
    accuracy, auc_score = evaluate_lightweight_model(model, val_gen, class_indices)
    
    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'lightweight_newcastle_detector.h5')
    model.save(final_model_path)
    
    # Final cleanup
    monitor_memory()
    
    print("\n" + "🎉" * 25)
    print("OPTIMIZED TRAINING COMPLETED!")
    print("🎉" * 25)
    log_message(f"Final Accuracy: {accuracy:.4f}")
    log_message(f"AUC Score: {auc_score:.4f}")
    log_message(f"Model saved: {final_model_path}")
    log_message(f"Peak RAM usage: {psutil.virtual_memory().percent:.1f}%")
    print("🎉" * 25)

if __name__ == "__main__":
    main()