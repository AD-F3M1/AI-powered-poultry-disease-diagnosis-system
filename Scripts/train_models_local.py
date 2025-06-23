# AI Poultry Disease Diagnosis - Local Dataset Training
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# Configuration
DATASET_PATH = "./dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # Reduced for faster training
MODEL_SAVE_PATH = "./models"

# Create models directory
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def log_message(message):
    """Log messages with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def check_dataset():
    """Check if dataset exists and is properly structured"""
    if not os.path.exists(DATASET_PATH):
        print("❌ Dataset folder not found!")
        print("Please create a 'dataset' folder with your images")
        return False
    
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    if len(classes) < 2:
        print("❌ Need at least 2 classes for training")
        return False
    
    total_images = 0
    for class_name in classes:
        class_path = os.path.join(DATASET_PATH, class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'))]
        total_images += len(images)
        
        if len(images) < 10:
            print(f"⚠️ Warning: Class '{class_name}' has only {len(images)} images")
    
    if total_images < 50:
        print(f"⚠️ Warning: Only {total_images} total images. Consider adding more for better results.")
    
    log_message(f"Dataset ready: {len(classes)} classes, {total_images} total images")
    return True

def create_data_generators():
    """Create data generators"""
    log_message("Creating data generators...")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    log_message(f"Training samples: {train_generator.samples}")
    log_message(f"Validation samples: {validation_generator.samples}")
    
    return train_generator, validation_generator

def build_simple_cnn(num_classes):
    """Build a simple CNN model for faster training"""
    log_message("Building CNN model...")
    
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_gen, val_gen, model_name):
    """Train the model"""
    log_message(f"Training {model_name} model...")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, f'{model_name}_best.h5'),
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f'{model_name}_history.png'))
    plt.show()

def evaluate_model(model, val_gen, class_indices, model_name):
    """Evaluate model"""
    log_message(f"Evaluating {model_name}...")
    
    val_gen.reset()
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Get class names
    class_names = list(class_indices.keys())
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print(f"\n{model_name} Results:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, f'{model_name}_confusion_matrix.png'))
    plt.show()
    
    # Newcastle-specific metrics
    newcastle_classes = [cls for cls in class_names if 'newcastle' in cls.lower()]
    if newcastle_classes:
        newcastle_name = newcastle_classes[0]
        log_message(f"Newcastle Disease Performance:")
        log_message(f"  Precision: {report[newcastle_name]['precision']:.3f}")
        log_message(f"  Recall: {report[newcastle_name]['recall']:.3f}")
        log_message(f"  F1-Score: {report[newcastle_name]['f1-score']:.3f}")
    
    return report['accuracy']

def main():
    """Main training function"""
    log_message("🐔 Starting AI Poultry Disease Diagnosis Training")
    
    # Check dataset
    if not check_dataset():
        return
    
    # Create data generators
    train_gen, val_gen = create_data_generators()
    class_indices = train_gen.class_indices
    num_classes = len(class_indices)
    
    log_message(f"Classes: {list(class_indices.keys())}")
    
    # Build and train model
    model = build_simple_cnn(num_classes)
    model.summary()
    
    # Train
    history = train_model(model, train_gen, val_gen, "CNN")
    
    # Plot results
    plot_history(history, "CNN")
    
    # Evaluate
    accuracy = evaluate_model(model, val_gen, class_indices, "CNN")
    
    # Save final model
    model.save(os.path.join(MODEL_SAVE_PATH, 'final_model.h5'))
    
    log_message(f"🎉 Training completed! Final accuracy: {accuracy:.3f}")
    log_message(f"Model saved in: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()