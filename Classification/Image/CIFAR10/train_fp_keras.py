import matplotlib.pyplot as plt
import copy
import random
import numpy as np
from keras import layers, optimizers
from tensorflow import keras
import tensorflow as tf
import os
# Skip GPU:0 which has less memory
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# Set random seeds for reproducibility


def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

# Enable memory growth to avoid OOM errors
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Enable mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set up multi-GPU strategy
strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')


class gRReLU(layers.Layer):
    def __init__(self, mean_scale=0.9, std_scale=0.1, mean_shift=0.1, std_shift=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mean_scale = tf.Variable(mean_scale, trainable=True)
        self.std_scale = tf.Variable(std_scale, trainable=True)
        self.mean_shift = tf.Variable(mean_shift, trainable=True)
        self.std_shift = tf.Variable(std_shift, trainable=True)

    def call(self, x, training=None):
        if training:
            eps_scale = tf.random.normal(tf.shape(x))
            eps_shift = tf.random.normal(tf.shape(x))
            scaled = x * eps_scale * self.std_scale + self.mean_scale
            shifted = self.std_shift * eps_shift + self.mean_shift
            return tf.where(x >= 0, x, scaled + shifted)
        else:
            slope = self.mean_scale + self.mean_shift
            return tf.where(x >= 0, x, slope * x)


class ResNet18(keras.Model):
    def __init__(self, num_classes=10, activation_type='relu'):
        super().__init__()

        self.activation_type = activation_type
        self.activation_fn = layers.ReLU() if activation_type == 'relu' else gRReLU()

        # Initial layers
        self.conv1 = layers.Conv2D(
            64, 7, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPooling2D(
            pool_size=3, strides=2, padding='same')

        # ResNet blocks
        self.block1_1 = self._build_block(64, stride=1)
        self.block1_2 = self._build_block(64)

        self.block2_1 = self._build_block(128, stride=2)
        self.block2_2 = self._build_block(128)

        self.block3_1 = self._build_block(256, stride=2)
        self.block3_2 = self._build_block(256)

        self.block4_1 = self._build_block(512, stride=2)
        self.block4_2 = self._build_block(512)

        # Output layers
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def _build_block(self, filters, stride=1):
        inputs = keras.Input(
            shape=(None, None, filters if stride == 1 else filters//2))

        # First conv block
        x = layers.Conv2D(filters, 3, strides=stride,
                          padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = self.activation_fn(x)

        # Second conv block
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # Shortcut connection
        if stride != 1:
            shortcut = layers.Conv2D(
                filters, 1, strides=stride, use_bias=False)(inputs)
            shortcut = layers.BatchNormalization()(shortcut)
        else:
            shortcut = inputs

        x = layers.Add()([shortcut, x])
        outputs = self.activation_fn(x)

        return keras.Model(inputs, outputs)

    def call(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x = self.maxpool(x)

        # ResNet blocks
        x = self.block1_1(x)
        x = self.block1_2(x)

        x = self.block2_1(x)
        x = self.block2_2(x)

        x = self.block3_1(x)
        x = self.block3_2(x)

        x = self.block4_1(x)
        x = self.block4_2(x)

        # Output
        x = self.avg_pool(x)
        return self.classifier(x)


def load_data():
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Convert to float32 and normalize
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Create smaller chunks of data
    chunk_size = 10000  # Process 10k images at a time
    train_chunks = []

    for i in range(0, len(x_train), chunk_size):
        end = min(i + chunk_size, len(x_train))
        x_chunk = x_train[i:end]
        y_chunk = y_train[i:end]

        # Create dataset from chunk
        chunk_dataset = tf.data.Dataset.from_tensor_slices((x_chunk, y_chunk))
        train_chunks.append(chunk_dataset)

    # Concatenate all chunks
    train_data = train_chunks[0]
    for chunk in train_chunks[1:]:
        train_data = train_data.concatenate(chunk)

    # Create test dataset
    test_chunks = []
    for i in range(0, len(x_test), chunk_size):
        end = min(i + chunk_size, len(x_test))
        x_chunk = x_test[i:end]
        y_chunk = y_test[i:end]
        chunk_dataset = tf.data.Dataset.from_tensor_slices((x_chunk, y_chunk))
        test_chunks.append(chunk_dataset)

    test_data = test_chunks[0]
    for chunk in test_chunks[1:]:
        test_data = test_data.concatenate(chunk)

    # Set up batching and prefetching
    batch_size = 16 * strategy.num_replicas_in_sync
    train_data = train_data.shuffle(1024).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Distribute the datasets
    train_data = strategy.experimental_distribute_dataset(train_data)
    test_data = strategy.experimental_distribute_dataset(test_data)

    return train_data, test_data


def inject_noise(model, alpha):
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D) or isinstance(layer, layers.Dense):
            weights = layer.get_weights()
            if len(weights) > 0:
                w = weights[0]
                noise = np.random.normal(0, alpha * np.abs(w), w.shape)
                weights[0] = w + noise
                layer.set_weights(weights)


def main():
    train_data, test_data = load_data()
    num_epochs = 200

    # Train models
    with strategy.scope():
        # ReLU model
        net_relu = ResNet18(activation_type='relu')
        net_relu.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # gRReLU model
        net_grrelu = ResNet18(activation_type='grrelu')
        net_grrelu.compile(
            optimizer=optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

    lr_schedule = optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=num_epochs
    )

    # Train ReLU model
    relu_history = net_relu.fit(
        train_data,
        epochs=num_epochs,
        validation_data=test_data,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'best_model_relu.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.LearningRateScheduler(lr_schedule)
        ]
    )

    # Train gRReLU model
    grrelu_history = net_grrelu.fit(
        train_data,
        epochs=num_epochs,
        validation_data=test_data,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                'best_model_grrelu.h5',
                save_best_only=True,
                monitor='val_accuracy'
            ),
            keras.callbacks.LearningRateScheduler(lr_schedule)
        ]
    )

    relu_best_acc = max(relu_history.history['val_accuracy']) * 100
    grrelu_best_acc = max(grrelu_history.history['val_accuracy']) * 100

    print("\nFinal Results:")
    print(f"ReLU Best Accuracy: {relu_best_acc:.2f}%")
    print(f"gRReLU Best Accuracy: {grrelu_best_acc:.2f}%")

    # Noise evaluation
    set_seed(42)

    alphas = np.arange(0, 1.6, 0.2)
    K = 10
    relu_accuracies_all = []
    grrelu_accuracies_all = []

    print("\nEvaluating models with different noise levels:")
    for alpha in alphas:
        relu_accuracies_alpha = []
        grrelu_accuracies_alpha = []

        print(f"Alpha={alpha:.1f}:")
        for k in range(K):
            # ReLU evaluation
            net_relu.load_weights('./best_model_relu.h5')
            net_relu_copy = keras.models.clone_model(net_relu)
            net_relu_copy.set_weights(net_relu.get_weights())
            inject_noise(net_relu_copy, alpha)
            results = net_relu_copy.evaluate(test_data, verbose=0)
            relu_acc = results[1] * 100
            relu_accuracies_alpha.append(relu_acc)

            # gRReLU evaluation
            net_grrelu.load_weights('./best_model_grrelu.h5')
            net_grrelu_copy = keras.models.clone_model(net_grrelu)
            net_grrelu_copy.set_weights(net_grrelu.get_weights())
            inject_noise(net_grrelu_copy, alpha)
            results = net_grrelu_copy.evaluate(test_data, verbose=0)
            grrelu_acc = results[1] * 100
            grrelu_accuracies_alpha.append(grrelu_acc)

            if (k + 1) % 10 == 0:
                print(f"  Completed {k + 1}/{K} evaluations")

        relu_accuracies_all.append(relu_accuracies_alpha)
        grrelu_accuracies_all.append(grrelu_accuracies_alpha)

        print(
            f"  ReLU Mean: {np.mean(relu_accuracies_alpha):.2f}% ± {np.std(relu_accuracies_alpha):.2f}%")
        print(
            f"  gRReLU Mean: {np.mean(grrelu_accuracies_alpha):.2f}% ± {np.std(grrelu_accuracies_alpha):.2f}%")

    # Plot results
    relu_means = np.array([np.mean(accs) for accs in relu_accuracies_all])
    relu_stds = np.array([np.std(accs) for accs in relu_accuracies_all])
    grrelu_means = np.array([np.mean(accs) for accs in grrelu_accuracies_all])
    grrelu_stds = np.array([np.std(accs) for accs in grrelu_accuracies_all])

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, relu_means, 'b-', label='ReLU')
    plt.fill_between(alphas, relu_means - relu_stds,
                     relu_means + relu_stds, color='b', alpha=0.2)
    plt.plot(alphas, grrelu_means, 'r-', label='gRReLU')
    plt.fill_between(alphas, grrelu_means - grrelu_stds,
                     grrelu_means + grrelu_stds, color='r', alpha=0.2)

    plt.xlabel('Noise Level (α)')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy vs Noise Level (with ±1 std)')
    plt.legend()
    plt.grid(True)
    plt.savefig('noise_robustness.png')
    plt.close()


if __name__ == "__main__":
    main()
