# train.py
import pandas as pd
import os
from tensorflow.keras import layers, models, applications, optimizers


# Настройки
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 8
IMAGE_DIR = 'images'
LABELS_CSV = 'labels.csv'
MODEL_OUT = 'model.h5'


# Загрузим метки
df = pd.read_csv(LABELS_CSV)
# Ожидается колонки: filename, dirty, damaged
assert {'filename','dirty','damaged'}.issubset(df.columns)


# Полные пути
df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))


# Создаём tf.data
file_paths = df['filepath'].values
labels = df[['dirty','damaged']].astype('float32').values


def preprocess(path, label):
image = tf.io.read_file(path)
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, IMG_SIZE)
image = image / 255.0
return image, label


dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
dataset = dataset.shuffle(len(file_paths)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_size = int(0.8 * len(file_paths))
train_ds = dataset.take(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = dataset.skip(train_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Модель: MobileNetV2 as backbone
base = applications.MobileNetV2(input_shape=(*IMG_SIZE,3), include_top=False, weights='imagenet')
base.trainable = False


inp = layers.Input(shape=(*IMG_SIZE,3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
# Два выхода — для dirty и damaged
out_dirty = layers.Dense(1, activation='sigmoid', name='dirty')(x)
out_damaged = layers.Dense(1, activation='sigmoid', name='damaged')(x)
model = models.Model(inputs=inp, outputs=[out_dirty, out_damaged])


model.compile(optimizer=optimizers.Adam(1e-4),
loss={'dirty':'binary_crossentropy', 'damaged':'binary_crossentropy'},
metrics={'dirty':'accuracy', 'damaged':'accuracy'})


model.summary()


history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


# Можно дообучить часть бэкбона
base.trainable = True
for layer in base.layers[:-20]:
layer.trainable = False
model.compile(optimizer=optimizers.Adam(1e-5),
loss={'dirty':'binary_crossentropy', 'damaged':'binary_crossentropy'},
metrics={'dirty':'accuracy', 'damaged':'accuracy'})
model.fit(train_ds, validation_data=val_ds, epochs=3)


# Сохраним
model.save(MODEL_OUT)
print('Saved model to', MODEL_OUT)
