

# Mushroom Classification Project

## Note

This project requires a large number of images to be scraped for training the classifier effectively. The plan is to scrape thousands of pictures to build a robust model. Currently, there is a bug in the scraping script that needs to be fixed. However, from previous scrappings, I have saved pictures that were used for training.

## Overview

This project focuses on scraping mushroom images from the iNaturalist website and using machine learning techniques to classify the mushrooms as edible or non-edible. The aim is to build a reliable model that can predict whether a mushroom in a new image is safe to eat.

## Libraries and Tools Used

- **Selenium**: For web scraping to collect mushroom images from iNaturalist.
- **Requests**: For downloading images from the web.
- **TensorFlow**: For building and training the machine learning model.
- **Keras**: A high-level API of TensorFlow used to simplify the construction and training of neural networks.
- **ImageDataGenerator**: A Keras utility for real-time data augmentation.
- **MobileNetV2**: A pre-trained model used as the base for transfer learning.
- **NumPy**: For numerical operations.

## Project Workflow

1. **Scraping Images from iNaturalist**:
    - We use the Selenium library to scrape images of mushrooms from the iNaturalist website.
    - The script navigates through the web pages, extracts image URLs, and downloads the images into specific directories for further processing.

    ```python
    import os
    import logging
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    import requests
    from time import sleep

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    def get_loaded_images():
        return driver.find_elements(By.CLASS_NAME, 'CoverImage')

    def download_images(url, download_folder, scroll_pause_time=50):
        driver.get(url)
        
        os.makedirs(download_folder, exist_ok=True)
        previous_image_count = 0

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            sleep(scroll_pause_time)

            current_images = get_loaded_images()
            current_image_count = len(current_images)

            if current_image_count == previous_image_count:
                break
            else:
                previous_image_count = current_image_count

        logger.info(f"Total images loaded: {current_image_count}")

        for i, image_div in enumerate(current_images):
            try:
                style_attribute = image_div.get_attribute('style')
                background_image_url = style_attribute.split('"')[1]

                response = requests.get(background_image_url)
                with open(os.path.join(download_folder, f"image_{i}.jpg"), 'wb') as img_file:
                    img_file.write(response.content)

                logger.info(f"Downloaded image {i+1}")

            except Exception as e:
                logger.error(f"Error downloading image {i+1}: {e}")

    galerina_url = "https://www.inaturalist.org/taxa/154735-Galerina-marginata/browse_photos"
    galerina_folder = "/Users/katka/Desktop/non_eatable_mushrooms"
    download_images(galerina_url, galerina_folder, scroll_pause_time=50)

    psilocybe_url = "https://www.inaturalist.org/taxa/328244-Psilocybe-cubensis/browse_photos"
    psilocybe_folder = "/Users/katka/Desktop/psilocybe_images"
    download_images(psilocybe_url, psilocybe_folder, scroll_pause_time=15)

    driver.quit()
    ```

2. **Data Preparation**:
    - The collected images are categorized into `edible` and `non_edible` mushrooms.
    - The dataset is then split into training and validation sets to ensure the model is evaluated correctly.

3. **Image Augmentation**:
    - Using Keras' `ImageDataGenerator`, we apply data augmentation techniques to increase the diversity of our training dataset. This helps in improving the robustness of our model.

4. **Building the Model**:
    ```python
    import os
    import random
    import shutil
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
    import numpy as np

    random.seed(42)
    tf.random.set_seed(42)

    edible_dir = "/Users/katka/Desktop/eatable_mushrooms"
    non_edible_dir = "/Users/katka/Desktop/non_eatable_mushrooms"
    new_images_dir = "/Users/katka/Desktop/new_mushrooms"

    train_dir = "/Users/katka/Desktop/train"
    validation_dir = "/Users/katka/Desktop/validation"

    categories = ['edible', 'non_edible']
    for category in categories:
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, category), exist_ok=True)

    def split_and_move_images(source_dir, train_dest_dir, val_dest_dir):
        images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
        random.shuffle(images)
        split_index = int(0.8 * len(images))
        train_images = images[:split_index]
        validation_images = images[split_index:]

        for img in train_images:
            shutil.copy(os.path.join(source_dir, img), os.path.join(train_dest_dir, img))
        for img in validation_images:
            shutil.copy(os.path.join(source_dir, img), os.path.join(val_dest_dir, img))

    split_and_move_images(edible_dir, os.path.join(train_dir, 'edible'), os.path.join(validation_dir, 'edible'))
    split_and_move_images(non_edible_dir, os.path.join(train_dir, 'non_edible'), os.path.join(validation_dir, 'non_edible'))

    img_size = (224, 224)
    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    train_edible_count = len(os.listdir(os.path.join(train_dir, 'edible')))
    train_non_edible_count = len(os.listdir(os.path.join(train_dir, 'non_edible')))
    val_edible_count = len(os.listdir(os.path.join(validation_dir, 'edible')))
    val_non_edible_count = len(os.listdir(os.path.join(validation_dir, 'non_edible')))

    print(f"Training set - Edible: {train_edible_count}, Non-edible: {train_non_edible_count}")
    print(f"Validation set - Edible: {val_edible_count}, Non-edible: {val_non_edible_count}")

    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                                   include_top=False,
                                                   weights='imagenet')

    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    ```

5. **Evaluation and Prediction**:
    ```python
    evaluation = model.evaluate(validation_generator)
    print("Validation Accuracy:", evaluation[1])

    model.save("/Users/katka/mushroom_classifier_model.h5")

    def predict_new_images(new_images_dir, model_path, img_size):
        new_images = os.listdir(new_images_dir)
        model = load_model(model_path)
        
        for image_file in new_images:
            image_path = os.path.join(new_images_dir, image_file)
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, 0)

            predictions = model.predict(img_array)
            predicted_class = "eatable" if predictions[0][0] < 0.5 else "non-eatable"
            
            print(f"Image: {image_file}, Predicted Class: {predicted_class}")

    predict_new_images("/Users/katka/Desktop/new_mushrooms", "/Users/katka/mushroom_classifier_model.h5", (224, 224))
    ```

## Results

The model achieves an accuracy of around 88% on the validation set, making it a useful tool for preliminary identification of edible mushrooms. However, always exercise caution and consult  when dealing with wild mushrooms.

## How to Use

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/mushroom-classification.git
    cd mushroom-classification
    ```

2. **Install Dependencies**:
    Ensure you have Python and pip installed, then run:
    ```bash
    pip install -r requirements.txt
    ```


3. **Train the Model**:
    Run the training script to train the model on the prepared dataset.
    ```python
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    ```

4. **Predict New Images**:
    Use the prediction script to classify new mushroom images.
    ```python
    predict_new_images("/Users/katka/Desktop/new_mushrooms", "/Users/katka/mushroom_classifier_model.h5", (224, 224))
    ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for any bugs or suggestions.
