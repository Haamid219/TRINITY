import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support, accuracy_score
from sklearn.manifold import TSNE
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add, Reshape, LeakyReLU, BatchNormalization, Flatten
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
import time
import json
import gc
import psutil
import os

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def process_data_in_chunks(filepath, chunk_size=100000):
    """Process data in chunks to manage memory usage."""
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # First pass: get feature columns and all unique labels
    print("First pass: Getting feature columns and unique labels...")
    unique_labels = set()
    feature_columns = None
    
    # Read through the file once to collect all unique labels
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        if feature_columns is None:
            feature_columns = [col for col in chunk.columns if col != 'Label']
        
        # No need to consolidate labels as dataset is already unified
        unique_labels.update(chunk['Label'].unique())
        
        del chunk
        gc.collect()
    
    # Initialize label encoder with all possible labels
    label_encoder = LabelEncoder()
    label_encoder.fit(list(unique_labels))
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    print(f"Label mapping: {label_mapping}")
    
    # Save feature order
    with open("/kaggle/working/feature_order.json", "w") as f:
        json.dump(feature_columns, f, indent=4)
    
    # Second pass: identify numeric columns and calculate feature medians
    print("Second pass: Identifying numeric columns and calculating feature medians...")
    numeric_columns = []
    categorical_columns = []
    feature_medians = {}
    
    # Process first chunk to determine column types
    first_chunk = next(pd.read_csv(filepath, chunksize=chunk_size))
    
    for col in feature_columns:
        # Check if column is numeric
        try:
            # If we can convert to numeric, it's numeric
            pd.to_numeric(first_chunk[col])
            numeric_columns.append(col)
            feature_medians[col] = []
        except (ValueError, TypeError):
            # If we can't convert to numeric, it's categorical
            categorical_columns.append(col)
    
    del first_chunk
    gc.collect()
    
    print(f"Identified {len(numeric_columns)} numeric columns and {len(categorical_columns)} categorical columns")
    print(f"Using only {len(numeric_columns)} numeric columns for training")
    
    # Continue processing chunks to calculate medians for numeric columns only
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunk = chunk[numeric_columns + ['Label']]  # Only include numeric columns
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Calculate median for numeric columns
        for col in numeric_columns:
            feature_medians[col].append(chunk[col].median())
        
        del chunk
        gc.collect()
    
    # Calculate final medians for numeric columns
    final_medians = {}
    for col in numeric_columns:
        final_medians[col] = np.nanmedian(feature_medians[col])
    
    np.save("/kaggle/working/feature_medians.npy", final_medians)
    np.save("/kaggle/working/numeric_columns.npy", numeric_columns)
    np.save("/kaggle/working/categorical_columns.npy", categorical_columns)
    
    # Third pass: process data in chunks and save to temporary files
    print("Third pass: Processing data in chunks...")
    temp_files = []
    chunk_index = 0
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        print(f"Processing chunk {chunk_index + 1}, memory usage: {get_memory_usage():.2f} GB")
        
        # Process chunk - only include numeric columns
        chunk = chunk[numeric_columns + ['Label']]
        
        # No need to consolidate labels as dataset is already unified
        
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Fill missing values with medians and handle data types
        for col in numeric_columns:
            chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            chunk[col] = chunk[col].fillna(final_medians[col])
            chunk[col] = chunk[col].astype(np.float64)
        
        # Encode labels
        chunk['Label'] = label_encoder.transform(chunk['Label'])
        
        # Save processed chunk
        temp_file = f"/kaggle/working/temp_chunk_{chunk_index}.npz"
        np.savez(temp_file, 
                X=chunk[numeric_columns].values,
                y=chunk['Label'].values)
        temp_files.append(temp_file)
        
        del chunk
        gc.collect()
        chunk_index += 1
    
    # Save label mapping to NPZ file
    np.savez("/kaggle/working/label_mapping.npz", label_mapping=label_mapping)
    
    # Load and concatenate chunks
    print("Loading and concatenating chunks...")
    X_chunks = []
    y_chunks = []
    
    for temp_file in temp_files:
        data = np.load(temp_file)
        X_chunks.append(data['X'])
        y_chunks.append(data['y'])
        os.remove(temp_file)  # Clean up temp file
        gc.collect()
    
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    
    print(f"Final data shape: {X.shape}")
    print(f"Memory usage after concatenation: {get_memory_usage():.2f} GB")
    
    return X, y, len(label_mapping), label_mapping, numeric_columns

def preprocess_data(filepath):
    """Load, clean, encode, and split dataset with all features."""
    print("Starting data preprocessing...")
    print("Using the unified dataset (classes already consolidated)...")
    
    X, y, num_classes, label_mapping, numeric_columns = process_data_in_chunks(filepath)
    
    print(f"Dataset has {num_classes} unique classes")
    
    # Save label mapping for later reference - now saved as NPZ in process_data_in_chunks
    # No longer saving as JSON
    
    # Scale features in chunks
    print("Scaling features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, "/kaggle/working/scaler.pkl")
    
    # Save processed data
    print("Saving processed data...")
    np.savez(
        "/kaggle/working/processed_numerical_features.npz", 
        X=X, 
        y=y,
        feature_names=np.array(numeric_columns)
    )
    
    print(f"Data shape with numerical features: {X.shape}")
    print(f"Number of unique classes: {num_classes}")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")
    
    return X, y, num_classes, label_mapping, numeric_columns

def display_class_distribution(y):
    """Display the distribution of classes."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
    
    plt.figure(figsize=(12, 6))
    plt.bar(unique_classes, class_counts)
    plt.xlabel("Class Label")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig("/kaggle/working/original_class_distribution.png")
    plt.show()

def create_tsne_visualization(X_data, y_data, class_names=None, perplexity=30, n_components=2):
    """Create t-SNE visualization of the data."""
    print("Performing t-SNE dimensionality reduction...")
    start_time = time.time()
    
    # Sample data if it's very large to make t-SNE faster
    max_samples = 10000
    if len(X_data) > max_samples:
        print(f"Sampling {max_samples} points for t-SNE visualization...")
        indices = np.random.choice(len(X_data), max_samples, replace=False)
        X_sample = X_data[indices]
        y_sample = y_data[indices]
    else:
        X_sample = X_data
        y_sample = y_data
    
    # Run t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    
    # Get unique classes
    unique_classes = np.unique(y_sample)
    
    # Create a colormap with distinct colors using the recommended approach
    cmap = plt.colormaps['tab10'].resampled(len(unique_classes))
    
    plt.figure(figsize=(12, 10))
    
    # Plot each class with a different color
    for i, cls in enumerate(unique_classes):
        idx = y_sample == cls
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                   c=[cmap(i)], marker='o', s=30, label=class_names[int(cls)] if class_names else f"Class {cls}",
                   alpha=0.7)
    
    plt.title('t-SNE Visualization of Network Traffic Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("/kaggle/working/tsne_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()

def build_transformer_model(sequence_length, feature_dim, num_classes, d_model=128, num_heads=4, num_layers=2):
    """Build transformer model for sequence classification."""
    # Input shape: (batch_size, sequence_length, feature_dim)
    inputs = Input(shape=(sequence_length, feature_dim))
    
    # Positional encoding
    positions = tf.range(start=0, limit=sequence_length, delta=1)
    positions = tf.expand_dims(positions, axis=1)  # (sequence_length, 1)
    positions = tf.cast(positions, dtype=tf.float32)
    
    # Create positional encodings
    angle_rads = positions / tf.pow(10000.0, 2 * tf.range(d_model, dtype=tf.float32) / d_model)
    angle_rads = tf.expand_dims(angle_rads, axis=0)  # (1, sequence_length, d_model)
    
    # Apply sine to even indices and cosine to odd indices
    sin_encodings = tf.sin(angle_rads[:, :, 0::2])
    cos_encodings = tf.cos(angle_rads[:, :, 1::2])
    pos_encoding = tf.concat([sin_encodings, cos_encodings], axis=-1)
    
    # Project input features to d_model dimensions
    x = Dense(d_model)(inputs)
    x = x + pos_encoding
    
    # Transformer blocks
    for _ in range(num_layers):
        # Multi-head attention block
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn_output = Dense(d_model * 4, activation='relu')(x)
        ffn_output = Dense(d_model)(ffn_output)
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global average pooling across the sequence dimension
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model

def plot_confusion_matrices(y_train, y_train_pred, y_test, y_test_pred, class_names=None):
    """Plot both training and testing confusion matrices side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Training confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("Actual Label")
    ax1.set_title("Training Confusion Matrix")
    
    # Testing confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel("Predicted Label")
    ax2.set_ylabel("Actual Label")
    ax2.set_title("Testing Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrices.png")
    plt.show()

def visualize_results(y_test, y_pred, y_train=None, y_train_pred=None, class_names=None, model_history=None):
    """Generate classification report, confusion matrix, ROC curve, and training vs validation graph."""
    
    # *1. Classification Report*
    test_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    print("Classification Report (Test):\n", classification_report(y_test, y_pred, target_names=class_names))
    
    # Save test classification report to file
    with open("/kaggle/working/classification_report_test.txt", "w") as f:
        f.write("Classification Report (Test):\n")
        f.write(classification_report(y_test, y_pred, target_names=class_names))
    
    # If training predictions are provided, save training classification report too
    if y_train is not None and y_train_pred is not None:
        train_report = classification_report(y_train, y_train_pred, target_names=class_names, output_dict=True)
        print("Classification Report (Train):\n", classification_report(y_train, y_train_pred, target_names=class_names))
        
        # Save training classification report to file
        with open("/kaggle/working/classification_report_train.txt", "w") as f:
            f.write("Classification Report (Train):\n")
            f.write(classification_report(y_train, y_train_pred, target_names=class_names))
    
    # *2. Confusion Matrices (Training and Testing) if training predictions are provided*
    if y_train is not None and y_train_pred is not None:
        plot_confusion_matrices(y_train, y_train_pred, y_test, y_pred, class_names)
    else:
        # Just testing confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.title("Testing Confusion Matrix")
        plt.tight_layout()
        plt.savefig("/kaggle/working/confusion_matrix.png")
        plt.show()
    
    # *3. ROC Curve (multi-class one-vs-rest)*
    num_classes = len(np.unique(y_test))
    if num_classes == 2:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color="blue")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line for reference
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.savefig("/kaggle/working/roc_curve.png")
        plt.show()
    
    # *4. Training vs Validation Loss & Accuracy (if model history is provided)*
    if model_history:
        plt.figure(figsize=(12, 5))

        # Loss Graph
        plt.subplot(1, 2, 1)
        plt.plot(model_history.history['loss'], label="Training Loss", color='blue')
        plt.plot(model_history.history['val_loss'], label="Validation Loss", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()

        # Accuracy Graph
        plt.subplot(1, 2, 2)
        plt.plot(model_history.history['accuracy'], label="Training Accuracy", color='blue')
        plt.plot(model_history.history['val_accuracy'], label="Validation Accuracy", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("/kaggle/working/training_vs_validation.png")
        plt.show()
    
    # *5. Save metrics to file*
    metrics = precision_recall_fscore_support(y_test, y_pred, average=None)
    accuracy = accuracy_score(y_test, y_pred)
    
    with open("/kaggle/working/metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {metrics[0]}\n")
        f.write(f"Recall: {metrics[1]}\n")
        f.write(f"F1 Score: {metrics[2]}\n")
    
    # Return metrics for potential additional use
    return {
        'accuracy': accuracy,
        'precision': metrics[0],
        'recall': metrics[1],
        'f1': metrics[2]
    }

def plot_class_distribution(y_data, class_names=None, title="Class Distribution"):
    """Plot class distribution."""
    plt.figure(figsize=(12, 6))
    counts = np.bincount(y_data.astype(int))
    plt.bar(range(len(counts)), counts, color='skyblue')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Samples')
    plt.title(title)
    if class_names:
        plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.tight_layout()
    plt.savefig("/kaggle/working/class_distribution.png")
    plt.show()

def create_sequences(X, y, sequence_length=5):
    """Create sequences from the data for transformer input."""
    print("Creating sequences...")
    print(f"Memory usage before sequence creation: {get_memory_usage():.2f} GB")
    
    sequences = []
    labels = []
    
    # Process in chunks to manage memory
    chunk_size = 10000
    for i in range(0, len(X) - sequence_length + 1, chunk_size):
        end_idx = min(i + chunk_size, len(X) - sequence_length + 1)
        chunk_sequences = []
        chunk_labels = []
        
        for j in range(i, end_idx):
            chunk_sequences.append(X[j:j + sequence_length])
            chunk_labels.append(y[j + sequence_length - 1])
        
        sequences.extend(chunk_sequences)
        labels.extend(chunk_labels)
        
        # Clean up
        del chunk_sequences, chunk_labels
        gc.collect()
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    print(f"Sequence shape: {sequences.shape}")
    print(f"Memory usage after sequence creation: {get_memory_usage():.2f} GB")
    
    return sequences, labels

def generate_consolidation_report(class_distribution_path):
    """Generate a report showing how classes were consolidated."""
    print("Class consolidation already performed. Skipping report generation.")
    return "Class consolidation already performed externally. See unify_classes.py for details."

def main():
    print("Starting main process...")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    print("=" * 80)
    print("NOTICE: Using the pre-unified dataset. No additional class consolidation needed.")
    print("NOTICE: This variant keeps ALL classes, regardless of sample count.")
    print("NOTICE: Only using numeric features for training (ignoring categorical features).")
    print("=" * 80)
    
    # Check for class distribution file in various locations
    possible_paths = [
        "class_distribution.txt",
        "/kaggle/input/dataset/class_distribution.txt",
        "/kaggle/working/class_distribution.txt"
    ]
    
    class_distribution_path = None
    for path in possible_paths:
        if os.path.exists(path):
            class_distribution_path = path
            break
            
    if class_distribution_path:
        print("Class distribution file found but consolidation report not needed (dataset pre-unified).")
    else:
        print("Class distribution file not found. Skipping consolidation report generation.")
    
    # Load and preprocess data with numeric features only
    # Check for the dataset in various locations
    possible_dataset_paths = [
        "/kaggle/input/dataset/CIC-IDS-2017(unified).csv",
        "CIC-IDS-2017(unified).csv",
        "/kaggle/working/CIC-IDS-2017(unified).csv"
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if not dataset_path:
        raise FileNotFoundError("Dataset file not found in any of the expected locations.")
    
    print(f"Using dataset at: {dataset_path}")
    X, y, num_classes, label_mapping, numeric_columns = preprocess_data(dataset_path)
    
    # Convert label mapping for better understanding
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Get class names from label mapping
    class_names = [inv_label_mapping.get(i, f"Class {i}") for i in range(num_classes)]
    
    print("=" * 80)
    print("Using Pre-Unified Dataset:")
    print("Dataset was pre-unified using unify_classes.py to merge '- Attempted' variants with parent classes")
    print(f"Number of unique classes: {num_classes}")
    print(f"Number of numeric features used: {len(numeric_columns)}")
    print("=" * 80)
    
    # Display the class distribution
    class_counts = np.bincount(y.astype(int))
    print("Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"Class {i} ({class_names[i]}): {count} samples")
    
    # Plot class distribution
    plot_class_distribution(y, class_names, "Class Distribution (All Classes)")
    
    # Create sequences from the data (no filtering)
    sequence_length = 5
    X_sequences, y_sequences = create_sequences(X, y, sequence_length)
    print(f"Shape of sequences: {X_sequences.shape}")
    
    # Clean up
    del X, y
    gc.collect()
    
    # Plot class distribution after sequencing
    plot_class_distribution(y_sequences, class_names, "Class Distribution After Sequencing")
    
    # Create t-SNE visualization of data
    print("Creating t-SNE visualization...")
    create_tsne_visualization(X_sequences.reshape(X_sequences.shape[0], -1), y_sequences, class_names)
    
    # Split into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_sequences, y_sequences, stratify=y_sequences, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")
    
    # Clean up
    del X_sequences, y_sequences, X_temp, y_temp
    gc.collect()
    
    # Save processed data
    np.savez(
        "/kaggle/working/processed_numeric_all_classes.npz", 
        X_train=X_train, X_val=X_val, X_test=X_test, 
        y_train=y_train, y_val=y_val, y_test=y_test,
        feature_names=np.array(numeric_columns)
    )
    
    # Learning rate schedule with warmup
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )
    
    # Build model for all classes
    model = build_transformer_model(
        sequence_length=sequence_length,
        feature_dim=X_train.shape[2],  # Use the actual feature dimension
        num_classes=num_classes
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Display class distribution (but don't use class weights)
    print("Class distribution (training without weights):")
    for i, count in enumerate(class_counts):
        print(f"Class {i} ({class_names[i]}): {count} samples")
    
    # Add early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
    )
    
    # Model checkpoint to save best model
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "/kaggle/working/best_model_numeric_only.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Train model without class weights
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # Save final model
    model.save("/kaggle/working/nids_transformer_numeric_only.keras")
    
    # Clean up
    # Keep X_train and y_train for generating training confusion matrix
    # del X_train, y_train, X_val, y_val
    del X_val, y_val
    gc.collect()
    
    # Generate predictions for test data
    y_test_pred = model.predict(X_test, batch_size=64)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    
    # Generate predictions for training data
    print("Generating predictions for training data...")
    y_train_pred = model.predict(X_train, batch_size=64)
    y_train_pred = np.argmax(y_train_pred, axis=1)
    
    # Visualize results including both training and testing confusion matrices
    metrics = visualize_results(
        y_test, y_test_pred, 
        y_train=y_train, y_train_pred=y_train_pred,
        class_names=class_names, 
        model_history=history
    )
    
    # Now it's safe to delete training data
    del X_train, y_train
    gc.collect()
    
    # Create t-SNE visualization of the model's embeddings
    print("Creating t-SNE visualization of model embeddings...")
    embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    test_embeddings = embedding_model.predict(X_test, batch_size=64)
    create_tsne_visualization(test_embeddings, y_test, class_names=class_names)
    
    print(f"Final memory usage: {get_memory_usage():.2f} GB")

if __name__ == "__main__":
    main() 