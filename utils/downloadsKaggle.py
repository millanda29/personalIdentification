import os
import pandas as pd
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Configuración
dataset_slug = "chiragsaipanuganti/utkface"  # Slug del dataset
base_dir = "datasets"  # Carpeta donde se descargará y procesará el dataset
output_dir = "output_yolo"  # Carpeta de salida para YOLO

# Inicializar la API de Kaggle
api = KaggleApi()
api.authenticate()

# Descargar el dataset
print("Descargando dataset desde Kaggle...")
api.dataset_download_files(dataset_slug, path=base_dir, unzip=True)
print(f"Dataset descargado y descomprimido en: {base_dir}")

# Diccionario para las clases (ajustado para valores numéricos)
class_map = {0: 0, 1: 1}  # Ajustar según los valores reales de 'gender'

# Crear carpetas para YOLO
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "images/test"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels/test"), exist_ok=True)

# Rutas del dataset
images_dir = os.path.join(base_dir, "Dataset/Images")  # Ajusta según la estructura real
index_dir = os.path.join(base_dir, "Dataset/Index")

# Verifica si los archivos CSV existen
if not os.path.exists(index_dir):
    print("Índices no encontrados, creando manualmente...")
    # Crear índices automáticamente
    files = os.listdir(images_dir)
    data = []
    for file in files:
        # Supongamos que los nombres de archivo contienen etiquetas como "age_gender_race_date.jpg"
        parts = file.split("_")
        if len(parts) >= 2:  # Validar formato esperado
            gender = 0 if parts[1] == "0" else 1  # Asumiendo que 0 es 'male' y 1 es 'female'
            data.append({"filename": file, "gender": gender, "filepath": os.path.join(images_dir, file)})

    # Si hay datos, procesar
    if data:
        df = pd.DataFrame(data)

        # Dividir en train, val y test
        train_df = df.sample(frac=0.8, random_state=42)
        temp_df = df.drop(train_df.index)
        val_df = temp_df.sample(frac=0.5, random_state=42)
        test_df = temp_df.drop(val_df.index)

        # Guardar los índices
        os.makedirs(index_dir, exist_ok=True)
        train_df.to_csv(os.path.join(index_dir, "Train.csv"), index=False)
        val_df.to_csv(os.path.join(index_dir, "Validation.csv"), index=False)
        test_df.to_csv(os.path.join(index_dir, "Test.csv"), index=False)
        print("Índices creados exitosamente.")
    else:
        print("No se encontraron imágenes válidas para crear los índices.")
        exit(1)  # Terminar el script si no se puede crear el CSV


# Función para procesar cada conjunto
def process_dataset(csv_file, split):
    # Verificar si el archivo CSV tiene datos
    if os.stat(csv_file).st_size == 0:
        print(f"El archivo CSV {csv_file} está vacío. Verifica su contenido.")
        return

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        image_name = row["filename"]
        label = row["gender"]

        # Imprimir para verificar el valor de 'label'
        print(f"Procesando imagen: {image_name}, género: {label}")

        # Si el valor de 'label' es inesperado, esto lo mostrará
        try:
            class_id = class_map[label]
        except KeyError:
            print(f"Error: '{label}' no está en el diccionario class_map")
            continue

        # Aquí se coloca la imagen en el directorio correcto dentro de output_yolo
        dst_image_path = os.path.join(output_dir, f"images/{split}", image_name)
        label_file = os.path.join(output_dir, f"labels/{split}", os.path.splitext(image_name)[0] + ".txt")

        # Ruta en la que la imagen está originalmente, dentro de 'Dataset/Images'
        src_image_path = os.path.join(images_dir, split.capitalize(), image_name)  # Ajuste aquí

        # Verificar si el archivo de imagen existe en la ruta indicada por 'filepath' en el CSV
        if os.path.exists(src_image_path):
            # Copiar la imagen a la carpeta correspondiente en output_yolo
            shutil.copy(src_image_path, dst_image_path)

            # Crear el archivo de etiqueta en formato YOLO
            with open(label_file, "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")


# Procesar cada conjunto (train, val, test)
process_dataset(os.path.join(index_dir, "Train.csv"), "train")
process_dataset(os.path.join(index_dir, "Validation.csv"), "val")
process_dataset(os.path.join(index_dir, "Test.csv"), "test")

print(f"Datos procesados y organizados en formato YOLO en: {output_dir}")
