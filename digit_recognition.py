import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 1. Загрузка и предобработка данных MNIST
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Нормализация: приводим значения пикселей к диапазону [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Изменение формы данных: добавляем канал (28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Преобразование меток в категориальный формат
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


# 2. Создание модели сверточной нейросети
def create_model():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# 3. Функция для предобработки пользовательского изображения
def preprocess_image(image_path):
    # Загрузка изображения в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Проверка, удалось ли загрузить изображение
    if img is None:
        raise ValueError("Не удалось загрузить изображение")

    # Инвертирование цветов (если фон черный, а цифра белая)
    img = cv2.bitwise_not(img)

    # Изменение размера до 28x28
    img = cv2.resize(img, (28, 28))

    # Нормализация
    img = img.astype("float32") / 255.0

    # Изменение формы для модели
    img = img.reshape(1, 28, 28, 1)

    return img


# 4. Основная функция для обучения и тестирования
def main():
    # Загрузка и предобработка данных
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    # Создание и обучение модели
    model = create_model()
    model.fit(
        x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test)
    )

    # Сохранение модели
    model.save("mnist_model.h5")
    print("Модель сохранена как 'mnist_model.h5'")

    # Оценка модели на тестовых данных
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Точность на тестовых данных: {test_accuracy*100:.2f}%")

    # Пример предсказания на пользовательском изображении
    try:
        image_path = input(
            "Введите путь к изображению с рукописной цифрой (например, 'digit.png'): "
        )
        processed_image = preprocess_image(image_path)

        # Загрузка сохраненной модели
        model = tf.keras.models.load_model("mnist_model.h5")

        # Предсказание
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction[0])

        print(f"Предсказанная цифра: {predicted_digit}")

        # Визуализация изображения
        plt.imshow(processed_image.reshape(28, 28), cmap="gray")
        plt.title(f"Предсказанная цифра: {predicted_digit}")
        plt.savefig("prediction_result.png")
        print("Результат предсказания сохранен как 'prediction_result.png'")

    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")


if __name__ == "__main__":
    main()
