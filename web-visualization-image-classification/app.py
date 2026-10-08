from flask import Flask, render_template, request, jsonify, send_file
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
import cv2
import numpy as np
import os
import imghdr
from datetime import datetime
import threading
import json

app = Flask(__name__)

# Global variables to track training status
training_status = {
    'is_training': False,
    'progress': 0,
    'logs': [],
    'model_path': None,
    'current_epoch': 0,
    'total_epochs': 0,
    'accuracy': 0,
    'loss': 0
}

class TrainingCallback(Callback):
    """Custom callback to track training progress"""
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        training_status['current_epoch'] = epoch + 1
        training_status['progress'] = ((epoch + 1) / self.total_epochs) * 80 + 20
        training_status['accuracy'] = logs.get('accuracy', 0)
        training_status['loss'] = logs.get('loss', 0)
        
        log_message = f"Epoch {epoch + 1}/{self.total_epochs} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}"
        if 'val_loss' in logs:
            log_message += f" - val_loss: {logs['val_loss']:.4f} - val_accuracy: {logs['val_accuracy']:.4f}"
        add_log(log_message, 'info')

def add_log(message, log_type='info'):
    """Add a log entry with timestamp"""
    training_status['logs'].append({
        'message': message,
        'type': log_type,
        'time': datetime.now().strftime('%H:%M:%S')
    })

def clean_dataset(data_dir, valid_exts=["jpg", "jpeg", "png", "bmp"]):
    """Remove invalid images from dataset"""
    removed_count = 0
    add_log('Starting dataset cleaning...', 'info')
    
    for image_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, image_class)
        if not os.path.isdir(class_dir):
            continue
            
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            try:
                tip = imghdr.what(image_path)
                if tip not in valid_exts:
                    add_log(f"Removing invalid file: {image_path}", 'warning')
                    os.remove(image_path)
                    removed_count += 1
            except Exception as e:
                add_log(f"Error processing {image_path}: {str(e)}", 'error')
                
    add_log(f"Cleaned {removed_count} invalid files", 'success')
    return removed_count

def count_images_and_classes(data_dir):
    """Count total images and classes in dataset"""
    total_images = 0
    classes = []
    
    for image_class in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, image_class)
        if not os.path.isdir(class_dir):
            continue
        classes.append(image_class)
        total_images += len([f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))])
    
    return total_images, len(classes), classes

def build_model(num_classes, image_size):
    """Build CNN model with dynamic number of output classes"""
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model_async(config):
    """Train model in background thread"""
    global training_status
    
    try:
        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['logs'] = []
        
        # Extract configuration
        data_dir = config['data_dir']
        epochs = config['epochs']
        batch_size = config['batch_size']
        train_split = config['train_split']
        val_split = config['val_split']
        test_split = config['test_split']
        image_size = config['image_size']
        
        add_log('Starting training process...', 'info')
        training_status['progress'] = 5
        
        # Validate dataset directory
        if not os.path.exists(data_dir):
            add_log(f"Error: Dataset directory '{data_dir}' not found!", 'error')
            training_status['is_training'] = False
            return
        
        # Clean dataset
        add_log('Scanning and cleaning dataset...', 'info')
        clean_dataset(data_dir)
        training_status['progress'] = 10
        
        # Count images and classes
        total_images, num_classes, class_names = count_images_and_classes(data_dir)
        add_log(f"Found {num_classes} classes with {total_images} images", 'success')
        add_log(f"Classes: {', '.join(class_names)}", 'info')
        training_status['progress'] = 15
        
        # Load dataset
        add_log('Loading dataset...', 'info')
        data = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            image_size=(image_size, image_size),
            batch_size=batch_size
        )
        training_status['progress'] = 20
        
        # Normalize data
        add_log('Normalizing images...', 'info')
        data = data.map(lambda x, y: (x / 255.0, y))
        
        # Calculate split sizes
        total_batches = len(data)
        train_size = int(total_batches * train_split)
        val_size = int(total_batches * val_split)
        test_size = int(total_batches * test_split)
        
        # Ensure we use all batches
        if train_size + val_size + test_size < total_batches:
            train_size += total_batches - (train_size + val_size + test_size)
        
        add_log(f"Split - Train: {train_size}, Val: {val_size}, Test: {test_size}", 'info')
        
        # Split dataset
        train = data.take(train_size)
        val = data.skip(train_size).take(val_size)
        test = data.skip(train_size + val_size).take(test_size)
        
        # Build model
        add_log('Building model architecture...', 'info')
        model = build_model(num_classes, image_size)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        add_log('Model compiled successfully', 'success')
        add_log(f"Model has {model.count_params():,} parameters", 'info')
        
        # Train model
        training_status['total_epochs'] = epochs
        add_log(f'Starting training for {epochs} epochs...', 'info')
        
        callback = TrainingCallback(epochs)
        
        model.fit(
            train,
            epochs=epochs,
            validation_data=val if val_size > 0 else None,
            callbacks=[callback],
            verbose=0
        )
        
        training_status['progress'] = 90
        
        # Evaluate on test set
        if test_size > 0:
            add_log('Evaluating on test set...', 'info')
            test_loss, test_accuracy = model.evaluate(test, verbose=0)
            add_log(f'Test accuracy: {test_accuracy:.4f}, Test loss: {test_loss:.4f}', 'success')
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f'trained_model_{timestamp}.h5'
        model.save(model_filename)
        training_status['model_path'] = model_filename
        
        add_log(f'Model saved as {model_filename}', 'success')
        add_log('Training completed successfully! 🎉', 'success')
        training_status['progress'] = 100
        
    except Exception as e:
        add_log(f'Error during training: {str(e)}', 'error')
        training_status['is_training'] = False
    finally:
        training_status['is_training'] = False

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    """Start training with given configuration"""
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    config = request.json
    
    # Validate configuration
    required_fields = ['data_dir', 'epochs', 'batch_size', 'train_split', 'val_split', 'test_split', 'image_size']
    for field in required_fields:
        if field not in config:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Start training in background thread
    thread = threading.Thread(target=train_model_async, args=(config,))
    thread.start()
    
    return jsonify({'message': 'Training started successfully'})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/download', methods=['GET'])
def download_model():
    """Download trained model"""
    if training_status['model_path'] and os.path.exists(training_status['model_path']):
        return send_file(training_status['model_path'], as_attachment=True)
    else:
        return jsonify({'error': 'No trained model available'}), 404

@app.route('/api/dataset/info', methods=['POST'])
def dataset_info():
    """Get information about a dataset"""
    data = request.json
    data_dir = data.get('data_dir')
    
    if not data_dir or not os.path.exists(data_dir):
        return jsonify({'error': 'Invalid dataset directory'}), 400
    
    total_images, num_classes, class_names = count_images_and_classes(data_dir)
    
    return jsonify({
        'total_images': total_images,
        'num_classes': num_classes,
        'class_names': class_names
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)