import cv2
import numpy as np
import tensorflow as tf
import torch

from ModeloTransformerTimmFactory import ModeloTransformerTimmFactory

# ============================================================
# CONFIGURACIÓN
# ============================================================

cat = ['carta_1', 'carta_2', 'carta_3', 'carta_4', 'carta_5']

# ============================================================
# CARGAR CNN PROPIA
# ============================================================

print("Cargando CNN propia...")

modelo_propio = tf.keras.models.load_model(
    "models/modeloA.keras",
    compile=False
)

print("CNN propia cargada")##

# ============================================================
# CARGAR SWIN TINY
# ============================================================

print("Cargando SwinTiny...")

modelo_swin, _ = ModeloTransformerTimmFactory.crear(
    nombreModelo="swin_tiny",
    categorias=cat,
    pesos="imagenet",
    congelar_base=False
)

modelo_swin.load_state_dict(
    torch.load(
        "models/modelo_swin_tiny_imagenet_finetuning.pth",
        map_location="cpu"
    )
)

modelo_swin.eval()

print("SwinTiny cargado")

# ============================================================
# FUNCIÓN CNN PROPIA
# ============================================================

def predecir_cnn(imagen):

    img = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2GRAY
    )

    img = cv2.resize(
        img,
        (30, 30)
    )

    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(
        img,
        axis=-1
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    pred = modelo_propio.predict(
        img,
        verbose=0
    )

    indice = np.argmax(pred)

    confianza = float(np.max(pred))

    return (
        cat[indice],
        confianza
    )

# ============================================================
# FUNCIÓN SWIN
# ============================================================

def predecir_swin(imagen):

    img = cv2.resize(
        imagen,
        (224, 224)
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32) / 255.0

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    tensor = torch.tensor(
        img,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        salida = modelo_swin(tensor)

        probs = torch.softmax(
            salida,
            dim=1
        )

        confianza, indice = torch.max(
            probs,
            dim=1
        )

    return (
        cat[indice.item()],
        confianza.item()
    )

# ============================================================
# ABRIR CÁMARA
# ============================================================

print("Iniciando cámara...")

cam = cv2.VideoCapture(
    1,
    cv2.CAP_AVFOUNDATION
)

if not cam.isOpened():

    print("No se pudo abrir la cámara 1")
    print("Prueba con 0")

    exit()

print("Cámara abierta correctamente")

# ============================================================
# LOOP PRINCIPAL
# ============================================================

texto_cnn = ""
texto_swin = ""

while True:

    ret, frame = cam.read()

    if not ret:
        break

    alto, ancho = frame.shape[:2]

    tam = 300

    x = (ancho - tam) // 2
    y = (alto - tam) // 2

    cv2.rectangle(
        frame,
        (x, y),
        (x + tam, y + tam),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "Coloque la carta dentro del cuadro",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "F = Capturar   ESC = Salir",
        (20, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        texto_cnn,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        texto_swin,
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Prediccion Tiempo Real",
        frame
    )

    tecla = cv2.waitKey(1) & 0xFF

    # ========================================================
    # CAPTURAR
    # ========================================================

    if tecla == ord("f"):

        roi = frame[
            y:y+tam,
            x:x+tam
        ]

        cv2.imwrite(
            "captura_objeto.jpg",
            roi
        )

        carta_cnn, conf_cnn = predecir_cnn(
            roi
        )

        carta_swin, conf_swin = predecir_swin(
            roi
        )

        texto_cnn = (
            f"Propio: {carta_cnn} "
            f"({conf_cnn*100:.2f}%)"
        )

        texto_swin = (
            f"SwinTiny: {carta_swin} "
            f"({conf_swin*100:.2f}%)"
        )

        print(texto_cnn)
        print(texto_swin)

    if tecla == 27:
        break

cam.release()
cv2.destroyAllWindows()