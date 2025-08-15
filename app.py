from google.colab import drive
drive.mount('/content/drive')

!pip install -q kaggle

from google.colab import files
files.upload()

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

! kaggle datasets list -s Tomography # kaggledaki tomografi ile ilgili veri setlerini gösterir

# Veri setini indirmek için gereken kod
!kaggle datasets download -d obulisainaren/retinal-oct-c8 -p /content/retinal_oct_c8 --unzip

from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import gradio as gr
from tensorflow.keras.preprocessing import image

# Veri yolları
train_dir = "/content/retinal_oct_c8/RetinalOCT_Dataset/RetinalOCT_Dataset/train"
val_dir = "/content/retinal_oct_c8/RetinalOCT_Dataset/RetinalOCT_Dataset/val"
test_dir = "/content/retinal_oct_c8/RetinalOCT_Dataset/RetinalOCT_Dataset/test"

image_size = (224, 224)
batch_size = 32
num_classes = len(os.listdir(train_dir))

# DataGenerator'lar, preprocess_input ile normalize ediliyor
train_gen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

val_test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    color_mode='rgb'
)

val_ds = val_test_gen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

test_ds = val_test_gen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    color_mode='rgb'
)

# Class weights hesapla
train_classes = train_ds.classes
print("Sınıf bazında örnek sayıları:", Counter(train_classes))

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)


# Learning rate schedule
initial_lr = 1e-3
epochs = 30
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_lr,
    decay_steps=epochs * (train_ds.samples // batch_size)
)


# Model oluşturma
def build_efficientnet_model(input_shape=(224,224,3), num_classes=8):
    base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    inputs = Input(shape=input_shape)
    x = base_model(preprocess_input(inputs), training=False)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model, base_model

model, base_model = build_efficientnet_model(input_shape=(224,224,3), num_classes=num_classes)




import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Callback'ler
checkpoint_cb = ModelCheckpoint(
    "efficientnet_best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Optimizörü sabit learning rate ile oluştur (ReduceLROnPlateau ile uyumlu)
base_lr = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

# Modeli compile et (loss/metrics'ini kendi problemine göre ayarla)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # ya da kullandığın uygun loss
    metrics=['accuracy']
)

# Fit
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb, early_stop, reduce_lr],
    verbose=1
)


 # En iyi modeli yükle
model = tf.keras.models.load_model("efficientnet_best_model.keras")

import tensorflow as tf

class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate
        # learning rate sabitse .numpy(), değilse (schedule) çağırarak al
        try:
            lr_value = lr.numpy()
        except:
            lr_value = lr(epoch)
        print(f"[LR] Epoch {epoch+1}: learning rate = {lr_value:.2e}")

# Fine tuning
base_model.trainable = True
fine_tune_at = len(base_model.layers) - 50
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base_model.layers[fine_tune_at:]:
    layer.trainable = True

# warmup tarzı: ilk 3 epoch 1e-4, sonra 1e-5
def scheduler(epoch, lr):
    if epoch < 3:
        return 1e-4
    return 1e-5

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb, early_stop, reduce_lr, PrintLR(), lr_callback],
    verbose=1
)

# En iyi modeli yükle
model = tf.keras.models.load_model("efficientnet_best_model.keras")

# Test değerlendirmesi
loss, acc = model.evaluate(test_ds, verbose=1)
print(f"Test Accuracy: {acc*100:.2f}%")

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_true = test_ds.classes
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=-1)
class_labels = list(test_ds.class_indices.keys())

print(classification_report(y_true, y_pred, target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap="Blues")
plt.title("Confusion Matrix")
plt.ylabel("Gerçek Sınıf")
plt.xlabel("Tahmin Edilen Sınıf")
plt.show()


import matplotlib.pyplot as plt

def plot_history(h, title_suffix=""):
    plt.figure()
    plt.plot(h.history['accuracy'], label='train_acc')
    plt.plot(h.history['val_accuracy'], label='val_acc')
    plt.title(f'Accuracy {title_suffix}')
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(h.history['loss'], label='train_loss')
    plt.plot(h.history['val_loss'], label='val_loss')
    plt.title(f'Loss {title_suffix}')
    plt.legend()
    plt.grid(True)

plot_history(history, "(initial)")
if 'history_finetune' in locals():
    plot_history(history_finetune, "(fine-tuning)")


!pip install gradio


import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- Varsayım: model global olarak yüklü olmalı ---
# model = tf.keras.models.load_model("efficientnet_best_model.keras")
class_labels = ["AMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "MH", "NORMAL"]

def preprocess_oct(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_disease(oct_image):
    if oct_image is None:
        return "", "", "Görüntü yüklenmedi. Lütfen bir OCT görüntüsü yükleyin.", ""
    try:
        x = preprocess_oct(oct_image)
        preds = model.predict(x)
        class_idx = int(np.argmax(preds, axis=-1)[0])
        confidence = float(np.max(preds))
        label = class_labels[class_idx] if class_idx < len(class_labels) else f"Class {class_idx}"
    except Exception as e:
        return "", "", f"Tahmin sırasında hata: {e}", ""
    info_map = {
        "NORMAL": "Görüntü normal, herhangi bir problem görünmüyor. 6 ayda bir göz kontrolü önerilir.",
        "AMD": "Age-related Macular Degeneration (Yaşa Bağlı Makula Dejenerasyonu.) Yaşlanma, genetik faktörler, sigara kullanımı, kötü beslenme, UV ışınlarına maruz kalma sebebiyle oluşur.",
        "CNV": "Koroidal neovaskülarizasyon (Yeni, anormal damar oluşumu.) Genellikle AMD’nin ileri evresiyle ilişkilidir, bazen yüksek miyopi nedeniyle oluşur.",
        "CSR": "Central serous retinopati (Retina altında sıvı birikimi.) Stres, kortizon kullanımı, hipertansiyon, uyku düzensizliği sebebiyle oluşur.",
        "DME": "Diyabetik makula ödemi (Diyabet kaynaklı sıvı sızıntısı.) Uzun süreli diyabet, kontrolsüz kan şekeri sebebiyle oluşur.",
        "DR": "Diyabetik retinopati (Kan damarlarında hasar ve sızıntı.) Diyabet süresinin uzun olması, yüksek tansiyon, yüksek kolesterol.",
        "DRUSEN": "Drusen (Retina altında biriken yağlı artıklar.) Yaşlanma, genetik yatkınlık sebebiyle oluşur.",
        "MH": "Makula deliği (Merkezi görüşte boşluk oluşumu.) Yaşlanma, vitreusun retinadan ayrılması, travma sebebiyle oluşur.",
    }
    solution_map = {
        "NORMAL": "Rutin takip. 6 ayda bir kontrole gelin.",
        "AMD": "Bir uzmana görünmeniz tavsiye edilir.Erken evrede vitamin ve mineral takviyesi (AREDS formülü), ileri evrede anti-VEGF enjeksiyonları,sağlıklı beslenme ve düzenli göz kontrolleri önerilir.",
        "CNV": "Hemen bir uzmana başvurmalısınız.Anti-VEGF enjeksiyonları (damar çoğalmasını engeller) ve fotodinamik tedavi önerilir.",
        "CSR": "Stres azaltma, gerekirse lazer/medikal takip.",
        "DME": "Kan şekeri kontrolü, gerekirse enjeksiyon tedavisi.",
        "DR": "Bir uzmana başvurmalısınız. Retina izleme, lazer veya medikal müdahale.",
        "DRUSEN": "Yaşamsal faktör yönetimi ve düzenli göz muayenesi.",
        "MH": " Hemen bir uzmana görünmelisiniz. Cerrahi (vitrektomi) değerlendirmesi önerilir.",
    }
    short_info = info_map.get(label, "Bu sınıf için özel bilgi yok.")
    solution = solution_map.get(label, "Uzman görüşü alın.")
    accuracy_display = f"{confidence*100:.1f}%"
    # Geri döndürülecek: label, doğruluk, kısa bilgi, çözüm önerisi
    return label, accuracy_display, short_info, solution

with gr.Blocks(css="""
    body { background: #eef7ff; }
    .title { font-size: 2.4rem; font-weight: 700; color: #1f4f8b; margin-bottom:5px; }
    .subtitle { font-size: 1rem; color: #1f4f8b; margin-top:0; }
    .btn-primary { background-color: #1f4f8b !important; color: white !important; }
    .btn-clear { background-color: #1f4f8b !important; color: white !important; }
    .card { border-radius: 12px; padding: 15px; background: white; box-shadow: 0 6px 18px rgba(31,79,139,0.08); }
    .small { font-size:0.85rem; color:#555; }
    .section-title { font-weight:600; margin-bottom:5px; }
""") as demo:

    info_screen = gr.Column(visible=True)
    upload_screen = gr.Column(visible=False)
    result_screen = gr.Column(visible=False)
    stored_image = gr.State(None)

    # --- 1. Giriş ---
    with info_screen:
        gr.Markdown("<div class='title'>OCT TOMOGRAFİ HASTALIK TAHMİNİ SİSTEMİ</div>")
        gr.Markdown("<div class='subtitle'>Yapay zekâ destekli retina tanı destek sistemi</div>")
        gr.Markdown("""
        ### Nasıl Kullanılır?
        1. Görüntü yükleme ekranına OCT Tomografi görüntünüzü yükleyiniz.
        2. Ardından 'Sonucu Göster' butonuna basın.
        3. Sonuç ekranında hastalık adı, ne olduğu, önerilen çözüm ve doğruluk yüzdesi çıkacak.
        4. Yeni görüntü yüklemek isterseniz sonuç ekranında 'Yeni Görüntü Yükle' butonuna basınız.
        """)
        gr.Markdown("<div class='small'>Bu sistem destekleyici amaçlıdır; kesin tanı için retina uzmanına danışın.</div>")
        btn_to_upload = gr.Button("Görüntü Yükle", elem_classes="btn-primary", size="lg")

    # --- 2. Yükleme ---
    with upload_screen:
        gr.Markdown("### OCT Görüntüsü Yükleyin", elem_classes="title")
        with gr.Row():
            with gr.Column(scale=2):
                oct_input = gr.Image(type="pil", label="OCT Görüntüsü")
                with gr.Row():
                    result_btn = gr.Button("Sonucu Göster", elem_classes="btn-primary")
                    clear_btn_upload = gr.Button("Resmi Sil", elem_classes="btn-primary")
            with gr.Column(scale=1):
                gr.Markdown("#### Yönergeler", elem_classes="subtitle")
                gr.Markdown("""
                - Net ve artefaktsız görüntü seçin.
                - JPEG/PNG formatı kullanın.
                - Gerekirse dışarıda kırpma yapıp dışarıda düzenleyip yükleyin.
                """)
        btn_back_to_info = gr.Button("Geri Dön")

    # --- 3. Sonuç ---
    with result_screen:
        gr.Markdown("### Tahmin Sonucu", elem_classes="title")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Yüklenen Görüntü")
                thumb = gr.Image(type="pil", label="Önizleme", interactive=False)
                clear_btn_result = gr.Button("Resmi Sil", elem_classes="btn-clear")
            with gr.Column(scale=2):
                gr.Markdown("<div class='section-title'>Hastalık Adı</div>")
                label_out = gr.Textbox(interactive=False)
                gr.Markdown("<div class='section-title'>Kısa Bilgi</div>")
                info_out = gr.Markdown()
                gr.Markdown("<div class='section-title'>Çözüm Önerisi</div>")
                solution_out = gr.Markdown()
                gr.Markdown("<div class='section-title'>Doğruluk</div>")
                accuracy_out = gr.Textbox(interactive=False)
        with gr.Row():
            btn_new_image = gr.Button("Yeni Görüntü Yükle", elem_classes="btn-primary")
            btn_home = gr.Button("Başlangıç Ekranına Dön")
        gr.Markdown("<div class='small'>Bu sistem yalnızca destekleyici amaçlıdır. Uzman muayenesi gerekir.</div>")

    # --- Etkileşimler ---
    btn_to_upload.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[info_screen, upload_screen]
    )
    btn_back_to_info.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[info_screen, upload_screen]
    )

    # Upload ekranında resmi sil
    clear_btn_upload.click(
        lambda: None,
        outputs=[oct_input]
    )

    # Tahmin yap ve sonucu göster
    def predict_and_store(img):
        # return disease outputs and also keep image for thumbnail
        label, accuracy, short_info, solution = predict_disease(img)
        return label, accuracy, short_info, solution, img

    result_btn.click(
        predict_and_store,
        inputs=[oct_input],
        outputs=[label_out, accuracy_out, info_out, solution_out, thumb]
    )
    result_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[upload_screen, result_screen]
    )

    # Sonuç ekranında resmi sil (thumbnail + geri yükleme)
    def clear_thumbnail():
        return "", "", "", "", None  # boşalt
    clear_btn_result.click(
        clear_thumbnail,
        inputs=[],
        outputs=[label_out, accuracy_out, info_out, solution_out, thumb]
    )

    # Yeni görüntü yükle (3 -> 2)
    btn_new_image.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[upload_screen, result_screen]
    )
    # Ana ekrana dön
    btn_home.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[info_screen, result_screen]
    )

    gr.Markdown("<footer style='margin-top:25px; font-size:0.8rem; color:#666;'>© 2025 Yapay Zekâ Destekli Göz Hastalığı Tahmin Sistemi</footer>")

demo.launch()








