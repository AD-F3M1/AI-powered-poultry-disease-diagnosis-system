# Optimized training for large dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

# Optimized configuration for large dataset
DATASET_PATH = "./dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 64  # Increased for large dataset
EPOCHS = 10      # Reduced since we have lots of data
MODEL_SAVE_PATH = "./models"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def log_message(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def create_optimized_generators():
    """Create optimized data generators for large dataset"""
    log_message("Creating optimized data generators...")
    
    # Less aggressive augmentation since we have lots of data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.15  # Smaller validation split since we have lots of data
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
    
    log_message(f"Training samples: {train_generator.samples:,}")
    log_message(f"Validation samples: {validation_generator.samples:,}")
    
    return train_generator, validation_generator

def build_robust_cnn(num_classes):
    """Build a more robust CNN for large dataset"""
    log_message("Building robust CNN model...")
    
    model = Sequential([
        # First block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Second block
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Third block
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        
        # Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Use a slightly lower learning rate for stability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_gen, val_gen):
    """Train the model with optimized callbacks"""
    log_message("Starting training...")
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=3,  # Reduced patience since we have lots of data
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODEL_SAVE_PATH, 'best_model.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            verbose=1
        )
    ]
    
    # Calculate steps per epoch
    steps_per_epoch = train_gen.samples // BATCH_SIZE
    validation_steps = val_gen.samples // BATCH_SIZE
    
    log_message(f"Steps per epoch: {steps_per_epoch}")
    log_message(f"Validation steps: {validation_steps}")
    
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plot_results(history):
    """Plot training results"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'training_results.png'), dpi=300)
    plt.show()

def evaluate_model(model, val_gen, class_indices):
    """Evaluate the model"""
    log_message("Evaluating model...")
    
    # Get predictions
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes
    
    # Get class names
    class_names = list(class_indices.keys())
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'), dpi=300)
    plt.show()
    
    # Newcastle Disease specific metrics
    newcastle_classes = [cls for cls in class_names if 'castle' in cls.lower()]
    if newcastle_classes:
        newcastle_name = newcastle_classes[0]
        log_message("NEW CASTLE DISEASE PERFORMANCE:")
        log_message(f"  Precision: {report[newcastle_name]['precision']:.4f}")
        log_message(f"  Recall: {report[newcastle_name]['recall']:.4f}")
        log_message(f"  F1-Score: {report[newcastle_name]['f1-score']:.4f}")
    
    return report['accuracy']

def main():
    """Main training function"""
    print("🐔" * 20)
    log_message("AI POULTRY DISEASE DIAGNOSIS - OPTIMIZED TRAINING")
    print("🐔" * 20)
    
    # Check GPU
    if tf.config.list_physical_devices('GPU'):
        log_message("GPU detected - training will be fast! 🚀")
    else:
        log_message("Using CPU - training will take longer ⏰")
    
    # Create data generators
    train_gen, val_gen = create_optimized_generators()
    class_indices = train_gen.class_indices
    num_classes = len(class_indices)
    
    log_message(f"Classes detected: {list(class_indices.keys())}")
    log_message(f"Number of classes: {num_classes}")
    
    # Build model
    model = build_robust_cnn(num_classes)
    
    # Show model summary
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    # Train model
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = train_model(model, train_gen, val_gen)
    
    # Plot results
    plot_results(history)
    
    # Evaluate
    accuracy = evaluate_model(model, val_gen, class_indices)
    
    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, 'final_poultry_model.h5')
    model.save(final_model_path)
    
    print("\n" + "🎉" * 20)
    log_message(f"TRAINING COMPLETED!")
    log_message(f"Final Accuracy: {accuracy:.4f}")
    log_message(f"Model saved: {final_model_path}")
    print("🎉" * 20)

if __name__ == "__main__":
    main()