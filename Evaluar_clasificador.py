import os
import shutil
from PIL import Image
import torch
from torchvision import models, transforms

# Configuración del dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformaciones para las imágenes nuevas
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Ruta al modelo y cargar el modelo entrenado
model_path = r"C:\Users\HP OMEN\Documents\Maestria\Primer semestre\Datos Ojos\Eyes test\codigo\Clasificador\resnet101_classifier.pth"
model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 clases: sanos, enfermos

# Cargar el modelo y mapearlo al dispositivo correcto
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit(1)

model = model.to(device)
model.eval()

# Carpetas de entrada y salida
input_folder = r"C:\Users\HP OMEN\Documents\Maestria\Primer semestre\Datos Ojos\Eyes test\codigo\Clasificador\base de datos\test"
output_folder = r"C:\Users\HP OMEN\Documents\Maestria\Primer semestre\Datos Ojos\Eyes test\codigo\Clasificador\evalua"
healthy_folder = os.path.join(output_folder, "sanos")
diseased_folder = os.path.join(output_folder, "enfermos")

# Crear carpetas de salida si no existen
os.makedirs(healthy_folder, exist_ok=True)
os.makedirs(diseased_folder, exist_ok=True)

# Función para predecir la clase de una imagen
def predict_image(image_path, model):
    try:
        image = Image.open(image_path).convert('RGB')  # Asegurar que sea RGB
        input_tensor = data_transforms(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            _, preds = torch.max(outputs, 1)

        return preds.item()
    except Exception as e:
        print(f"Error procesando la imagen {image_path}: {e}")
        return None

# Clasificar las imágenes y moverlas a las carpetas correspondientes
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
total_images = len([f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)])

for idx, filename in enumerate(os.listdir(input_folder), start=1):
    file_path = os.path.join(input_folder, filename)

    if os.path.isfile(file_path) and filename.lower().endswith(valid_extensions):
        try:
            # Predecir la clase de la imagen
            predicted_class = predict_image(file_path, model)

            # Mover la imagen a la carpeta correspondiente si se clasificó con éxito
            if predicted_class is not None:
                if predicted_class == 1:  # Clase 0: sanos
                    shutil.move(file_path, os.path.join(healthy_folder, filename))
                elif predicted_class == 0:  # Clase 1: enfermos
                    shutil.move(file_path, os.path.join(diseased_folder, filename))

                print(f"[{idx}/{total_images}] {filename} clasificada como {'sanos' if predicted_class == 0 else 'enfermos'}")
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

print("Clasificación y organización completadas.")