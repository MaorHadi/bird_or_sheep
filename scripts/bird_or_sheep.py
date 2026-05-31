import tensorflow as tf


classes = ["bird", "sheep"]

img_size = (64, 64)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=42
)

class_names = train_ds.class_names


AUTOTUNE = tf.data.AUTOTUNE

normalize = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalize(x), y))
val_ds   = val_ds.map(lambda x, y: (normalize(x), y))

train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)


print("Classes:", class_names)

counts = {name: 0 for name in class_names}

for _, labels in train_ds.unbatch():
    counts[class_names[int(labels)]] += 1
print("Sample counts:", counts)

total = sum(counts.values())

class_weight = {i: total / (len(counts) * counts[name]) 
                for i, name in enumerate(class_names)}

print("Class weights:", class_weight)


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=img_size + (3,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])

model.compile(
    optimizer='adam',
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    class_weight=class_weight
)

model.save('bird_sheep_model.keras')
print("\n✅ Model saved as 'bird_sheep_model.keras'")