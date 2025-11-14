import kagglehub
import pandas as pd
import os

def load_data():
    """
    Загружает данные из Kaggle Fashion Product Images Dataset
    """
    try:
        # Новый API для kagglehub
        path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    except AttributeError:
        # Альтернативный метод
        try:
            path = kagglehub.datasets.download("paramaggarwal/fashion-product-images-dataset")
        except:
            # Если ничего не работает, попробуем найти уже скачанный датасет
            home_dir = os.path.expanduser("~")
            possible_paths = [
                os.path.join(home_dir, ".cache/kagglehub/datasets/paramaggarwal/fashion-product-images-dataset"),
                os.path.join(home_dir, ".kaggle/datasets/paramaggarwal/fashion-product-images-dataset"),
            ]
            
            path = None
            for p in possible_paths:
                if os.path.exists(p):
                    # Найдем самую свежую версию
                    versions = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
                    if versions:
                        latest = sorted(versions)[-1]
                        path = os.path.join(p, latest)
                        break
            
            if path is None:
                raise Exception("Не удалось найти датасет. Скачайте его вручную с Kaggle.")

    print("Path to dataset files:", path)

    # Пытаемся найти CSV файлы
    csv_file = None
    image_file = None
    
    # Возможные пути к файлам
    possible_csv_paths = [
        os.path.join(path, "fashion-dataset/styles.csv"),
        os.path.join(path, "styles.csv"),
    ]
    
    possible_image_paths = [
        os.path.join(path, "fashion-dataset/images.csv"),
        os.path.join(path, "images.csv"),
    ]
    
    for csv_path in possible_csv_paths:
        if os.path.exists(csv_path):
            csv_file = csv_path
            break
    
    for img_path in possible_image_paths:
        if os.path.exists(img_path):
            image_file = img_path
            break
    
    if csv_file is None:
        raise FileNotFoundError(f"Не найден файл styles.csv. Проверьте путь: {path}")
    
    # Загружаем CSV
    try:
        data = pd.read_csv(csv_file, on_bad_lines='skip')
        # images.csv может отсутствовать - это нормально
        if image_file and os.path.exists(image_file):
            images = pd.read_csv(image_file, on_bad_lines='skip')
        else:
            # Создаем пустой DataFrame если images.csv нет
            images = pd.DataFrame()
    except TypeError:
        # Для старых версий pandas
        data = pd.read_csv(csv_file, error_bad_lines=False)
        if image_file and os.path.exists(image_file):
            images = pd.read_csv(image_file, error_bad_lines=False)
        else:
            images = pd.DataFrame()
    except Exception as e:
        print("Error parsing CSV files:", e)
        data = pd.DataFrame()
        images = pd.DataFrame()

    print("Data loaded successfully!")
    print(f"Data shape: {data.shape}")
    print(data.head())
    
    return data, images, path  # Возвращаем также путь к датасету