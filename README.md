# ANPR: Automatic Number Plate Recognition

This directory contains all the materials developed for a Computer Vision project as part of a course in the Data Engineering degree at the Universitat Autònoma de Barcelona (UAB). Below is a description of the folder structure and the content of each of its components.

## Folder Structure
Localization: Contains different versions of the code used to develop the license plate localization system during the project.
Segmentation: Includes the various versions implemented for the character segmentation process of the license plates.
Recognition: Holds the versions of the code used to recognize the characters (letters and numbers) of the plates after segmentation.
Each folder contains the files that evolved throughout the process until reaching the final solution.

### Final Version
Inside the FINAL VERSION folder, the final codes developed for each process can be found:

    - Localization.py: Final code for the license plate localization system.
    - Segmentation.py: Code implementing the complete character segmentation process once the plate is localized.
    - Recognition.py: Code to load segmented images and perform the corresponding recognition (letters or numbers).
    - Models_Localization_Recognition Folder
    - The Models_Localization_Recognition folder contains the model files used for the license plate localization and recognition system.

### General Program
The file All Toghether.py is the main program that integrates all the previous processes. This code imports the scripts Localization.py, Segmentation.py, and Recognition.py to execute the entire system from a car image and produce the final recognized license plate.


## Introduction
This project focuses on the development of an "Automatic Number Plate Recognition" (ANPR) system, which aims to automatically detect and recognize license plates in vehicle images. The primary goal is to facilitate access to information about vehicles and their owners. ANPR systems are primarily used for access control in parking areas, although their main application is to issue fines for traffic violations.

Throughout this project, we will explore how these systems work by following four key steps: acquiring images, locating the license plate, segmenting the characters, and recognizing them. The main challenges lie in accurately locating and identifying license plates using computer vision and machine learning techniques.

The primary issues that need to be addressed in this project include effectively locating license plates in complex images, precisely segmenting the characters, and accurately identifying them.

## Objectives
The main goal of this challenge is to develop an Automatic Number Plate Recognition (ANPR) system capable of accurately detecting, identifying, and recognizing vehicle license plates in images. To achieve this goal, several specific tasks need to be completed, which include four main steps:

1. **Image Acquisition:** Capturing vehicle images from different angles.
2. **Localization:** Identifying the part of the image containing the license plate.
3. **Segmentation:** For each license plate, separating the characters from the background.
4. **Recognition:** Identifying each character to determine the corresponding license plate.

This project focuses on Spanish license plates, which follow a specific format of four numbers followed by three letters.

## Methodology
The workflow for the license plate detection project, consists of four main steps:

1. **Acquisition:** The first step involves capturing images of different vehicles. These images are stored in a database and can originate from various sources.
2. **Localization:** Once the images are acquired, they are processed to locate the license plate within each image. This step utilizes image processing techniques to detect the portion of the image containing the license plate.
3. **Segmentation:** After identifying the license plate, the characters on the plate are separated from the background of the image. This step ensures that the characters are clearly defined for further identification.
4. **Recognition:** Finally, machine learning techniques are employed to recognize each of the segmented characters, yielding the final identified license plate value.

This process ensures the automatic detection and recognition of license plates, aiding in vehicle access control or traffic fine management.

## Acquisition of Photographs

Photographs were obtained from various sources, such as captures taken with mobile cameras, images provided by our tutor [1], or from online databases such as Kaggle [2]. These photographs were partially used to create our own test set, while the rest were used to build the training sets for the models employed.

### Method 1: Manual Photograph Capture

To collect license plate images, we used our mobile phones to capture photographs of parked vehicles in a parking lot. Images were taken from different angles to ensure a variety of perspectives, including plates under different lighting conditions and distances. Once the photographs were captured, they were transferred to our computers to be incorporated into the working database, complementing them with images from other sources.

### Method 2: Online Data Collection

In addition to manually captured images, we expanded our database by combining it with the dataset provided by our tutor [1]. Additionally, we used an extended dataset from the Kaggle platform [3], specifically utilized for training the YOLOv5 model for localization. This process allowed us to access a broader collection of license plate images, increasing the diversity and quantity of material available for our project. Some images from this dataset were incorporated into our own test set (along with manually collected images and those provided by the tutor).

## Localization

For license plate localization, a YOLOv5 Neural Network was used, an advanced model widely applied in computer vision tasks. This model processes the original captured images to accurately identify the region containing the vehicle’s license plate. By performing detailed image analysis, YOLOv5 efficiently locates the license plate's characters, highlighting their exact position. The final outcome of this process is a set of images where license plates from the original images are clearly localized and prepared for the next processing stage.

### YOLOv5 Training

The YOLOv5 network was trained using a specific Kaggle database [3], specifically designed for this purpose. This database contains images and labels that allowed us to train our network to locate license plates in any image, regardless of angle, lighting, or other environmental factors.

### Part 1: License Plate Detection with YOLOv5

YOLOv5, a state-of-the-art object detection model, was used for localizing license plates in the images. This model is notable for its speed and accuracy [5]. YOLOv5 allows efficient localization of plates even under low-light conditions or when plates are partially obstructed.

#### Advantages of YOLOv5:
- **Speed:** YOLOv5 is extremely fast and capable of real-time object detection.
- **Accuracy:** Despite its speed, the model maintains high accuracy in detection, which is critical for confidently identifying license plates.
- **Flexibility:** It can be trained to detect any type of object, making it adaptable to various scenarios.
- **Versatility:** It can detect multiple objects of the same type within a single image.

#### Workflow:
1. **Model Training:** Train the YOLOv5 model using the Kaggle dataset to recognize license plates.
2. **Model Loading:** Use the custom-trained YOLOv5 model specifically adjusted to detect vehicle plates.
3. **Image Loading:** Load the image to be analyzed using the OpenCV library.
4. **Object Detection:** Apply detection on the image, returning the bounding box coordinates of the detected plate.
5. **Best Detection Filtering:** Select the detection with the highest confidence from all possible detections.
6. **Bounding Box Visualization:** Draw the bounding box around the detected plate.
7. **Bounding Box Cropping:** Crop the bounding box with a margin.
8. **Coordinate Retrieval:** Use the bounding box coordinates to crop the detected plate.
9. **Repeat Localization:** Reapply localization to further refine and eliminate any non-plate elements in the image.

### Part 2: Image Processing and Alignment

Once the plate is localized, additional processing is conducted to improve its quality before segmenting the characters. This process involves converting the image to HSV format, extracting the V (brightness) channel, and applying filters to enhance the contours of the characters. A rotation is also applied to ensure the characters are horizontally aligned.

### Advantages of this method:
- **Image Clarification:** The V channel works directly with brightness information, facilitating character segmentation.
- **Automatic Rotation:** The algorithm detects and corrects any plate tilt, ensuring the characters are perfectly aligned for subsequent recognition.

### Workflow:
1. **V Channel Extraction:** Convert the image to HSV format and extract the brightness channel.
2. **Contour Detection:** Apply the Canny edge detector to identify the edges of the characters.
3. **Image Alignment:** Calculate the minimum rotation necessary to align the characters and straighten the plate.
4. **Second YOLOv5 Localization:** Reapply YOLOv5 localization to further eliminate remaining car fragments around the plate.
5. **Processed Image Retrieval:** Obtain the final color image with the localized and straightened plate.

## Segmentation

License plate segmentation employs various image processing techniques, such as HSV color space usage, contour detection, and thresholding. These methods allow for precise separation of characters from the background. Once segmented, characters are individually cropped and correctly ordered, preparing them for the recognition phase.

### Part 1: Contour Detection

Contour detection is a fundamental image processing method that identifies object shapes and boundaries within an image. The `cv2.findContours` function from OpenCV is used to detect the contours of the plate's characters.

#v Recognition

The recognition phase employs two machine learning models tailored to different types of characters. A convolutional neural network (CNN) is used to recognize numeric characters, leveraging its efficiency in image processing and pattern recognition. For letters, a Support Vector Machine (SVM) model is applied due to its robustness in classification tasks.

Once the CNN and SVM models have recognized the characters, they are combined to form a complete license plate string in a readable format, enabling easy vehicle identification.

## Experimental Design

### Dataset Description
For each phase of the project (Localization, Segmentation, and Recognition), specific datasets were utilized, tailored to the corresponding tasks. A global dataset with original vehicle images captured under varied conditions was used to simulate real-world scenarios and evaluate the system's overall performance. This dataset includes 47 different images.

### 3.1.1 Localization

In the localization phase, we used a Kaggle dataset specifically designed for object detection [3], particularly for license plates. This dataset consists of two folders: one containing images of vehicles and another with "labels" files that include the coordinates of the license plates for each image. 

The "labels" files contain the class number and the coordinates `x1`, `y1`, `x2`, `y2`. An important detail to note is that the vehicles in this dataset are not of Spanish origin. This is because, at this stage of the project, our focus is not on recognizing the letters of the license plates; instead, we aim to detect the license plate itself as accurately as possible.

### 3.1.2 Segmentation

In this phase of the project, no specific model was developed or trained for segmentation. Instead, we worked directly with the images generated after applying the localization system to the photographs in our test database, which consists of 47 images.

The images used (derived from the localization phase applied to our test set) are real representations of Spanish vehicle license plates, enabling us to validate the effectiveness of our segmentation method. This approach simplifies the optimization of detection and segmentation processes without the need to split the data into training and test sets, as the primary goal is to improve segmentation accuracy using images that have already been localized.

### 3.1.3 Recognition

For the dataset used in the recognition phase, the training images were artificially generated using the Spanish License Plate Font [4]. For each character, 45 images were created by applying rotations at angles of -12°, -9°, -6°, -3°, 0°, 3°, 6°, 9°, and 12°, as well as dilation and erosion processes with 3x3 and 5x5 kernels. This set of transformations simulates variations that may appear in segmented images, such as slight tilts or changes in character thickness, thereby enhancing the model's ability to generalize and recognize characters in real-world conditions. In total, we generated 1,395 images, 450 of which correspond to numbers and 945 to letters.

For the test set, the 47 images from our custom test dataset were used, separating the characters from each license plate individually, resulting in a total of 336 images (192 numbers and 144 letters). These images were derived directly from the segmentation phase, allowing us to evaluate the performance of the recognition models under real-world conditions and ensure that they can correctly recognize characters after the license plate segmentation process.

### 3.2 Experiments and Metrics

#### 3.2.1 Localization
As previously mentioned, the YOLOv5 model was utilized in this stage of the project. A single model was trained over 50 epochs. Other hyperparameters used were the default settings of YOLOv5, which include:

| **Hyperparameter**          | **Value**  | **Explanation**                                                                                   |
|-----------------------------|------------|---------------------------------------------------------------------------------------------------|
| **Initial Learning Rate (lr0)** | 0.01       | Indicates the magnitude with which weights are adjusted during training.                         |
| **Momentum**                | 0.937      | Used to smooth gradient updates, helping accelerate learning.                                    |
| **Weight Decay**            | 0.0005     | Helps prevent overfitting by penalizing the model's weights, keeping them small and improving generalization. |
| **Epochs**                  | 3          | Number of epochs during which the learning rate gradually increases to avoid premature model convergence. |
| **Box Weight**              | 0.05       | Controls the error for bounding box predictions.                                                 |
| **Object Weight**           | 1.0        | Indicates the error for object predictions, prioritizing positive detections.                    |
| **Optimizer**               | SGD        | The optimizer responsible for updating the model weights.                                        |
| **Batch Size**              | 16         | Number of images processed at a time in each training iteration.                                 |
| **Input Image Size**        | 640        | Dimensions of the input images used during training.                                             |

#### Metrics
YOLOv5 generates a large number of outputs, but the most relevant for our purposes are **precision**, **recall**, and **loss**:

- **Precision**: High precision in license plate detection indicates that most predictions are correct. This is essential to avoid false positives (e.g., incorrect detections that are not license plates).
- **Recall**: High recall ensures that the model detects all the license plates in the images, even under challenging conditions, avoiding the loss of important information.
- **Loss**: This metric measures the model's accuracy by comparing predictions to actual labels. A lower loss indicates the model is fitting well to the training data.

Additionally, metrics that relate these values, such as the **Precision-Recall Curve (PR Curve)**, were used. The PR Curve combines precision and recall into a single graph, showing their relationship across different thresholds. This is particularly useful for imbalanced classes, such as license plate detection. A curve approaching the top-right corner indicates good model performance, with high precision and recall simultaneously.

#### Techniques Used
1. **Training the Model**: The model was trained using the Kaggle dataset [3].
2. **Loading the Model**: The best epoch weights of the trained YOLOv5 model were loaded.
3. **Image Loading**: The image for license plate detection was loaded using the OpenCV library.
4. **Object Detection**: The model applied detection to the image, returning the bounding box coordinates of detected plates. It can detect more than one license plate in a single image.
5. **Filtering the Best Detection**: Among all detections, the one with the highest confidence score was selected. This ensures the most reliable detection is chosen.
6. **Visualization of the Bounding Box**: A rectangle was drawn around the detected license plate.
7. **Cropping the Bounding Box**: The bounding box was cropped from the image with a margin. A margin of 10 was applied in the first localization step, and 50 in the second, to avoid characters touching the edges of the image and causing distortions during rotations or cropping characters.
8. **Extraction of the V Channel**: The image was converted to HSV format, and the brightness (V channel) was extracted. This step improves character detection, as the inclination is used to rotate the license plate.
9. **Contour Detection**: Thresholding was applied, followed by the Canny edge detector, to identify the edges of the characters. The Canny detector threshold was set between 50 and 150, ensuring precise detection of character edges. Additionally, a minimum area of 100 was established, filtering out smaller objects or noise.
10. **Image Alignment**: The minimum rotation needed to align the characters was calculated, and a transformation was applied to straighten them. This rotation was applied to the cropped license plate image (in color), not the thresholded one. For example, slightly tilted plates with straight characters do not require this adjustment.
11. **Second YOLOv5 Application**: A second localization was applied to the cropped license plate image with a margin of 50. The purpose was to eliminate most of the surrounding vehicle and retain only the license plate. If the license plate was not detected in this second step, the image remained unchanged.
12. **Processed Image Retrieval**: A processed image with rectified characters was obtained, ready for segmentation and recognition steps.


### 3.2.2 Segmentation

During this stage of the project, tests were conducted to incorporate new techniques aimed at improving the detection of character contours in license plate images. The main objective was to enhance the precision and effectiveness of the segmentation process, ensuring that characters were accurately identified and the background was effectively excluded.

#### Techniques Used
1. **Brightness Adjustment**: The brightness of the images was adjusted using a contrast factor (`brillo_alpha = 1.46`) and a brightness value (`brillo_beta = 70`). This adjustment was iteratively fine-tuned to achieve optimal differentiation between characters and the background, improving contour detection.

2. **Conversion to HSV Color Space**: Images were converted to the HSV color space to focus on the value (V) channel, which highlights bright regions and transforms them into darker ones. This improved character contour detection by enhancing their definition.

3. **Thresholding**: The `cv2.threshold` function was used to create a binary image, where pixels above a threshold of 230 were converted to black, and those below to white. This helped separate the characters from the background for further analysis.

4. **Contour Detection with Canny**: The Canny edge detection algorithm was applied to identify character edges more precisely. Adjustable parameters allowed for experimentation, and final thresholds were set to 100 (lower) and 500 (upper) to optimize results. This ensured better contour detection, especially in cases where initial thresholding did not yield clear edges.

5. **Dilation**: A dilation operation was performed using a rectangular kernel of size (3, 3). This step connected fragmented character contours without losing significant structural details, ensuring continuity.

6. **Contour Filtering**: Contours were filtered to select only those corresponding to characters. Using `cv2.findContours`, contours were extracted from the dilated image, and parameters such as height (`h > 50`) and width-to-height ratio (`0.1 < w / h < 1`) were applied to isolate valid character boundaries. Additional techniques included margin adjustments of 5 pixels to exclude contours near image edges and hierarchical detection for nested contours like those in the letter 'P'.

7. **Character Extraction**: After filtering, characters were segmented individually using bounding rectangles. This step ensured that each character was isolated for further processing.

8. **Character Ordering**: Characters were ordered based on their X-coordinates in the original image, ensuring they appeared in the correct sequence as on the license plate. This guarantees accurate processing in downstream steps.

9. **Adjustment to 7 Characters**: Since Spanish license plates consist of 7 characters, adjustments were made to ensure this number:
   - **More than 7 Characters**: Smaller contours were discarded until only 7 remained.
   - **Fewer than 7 Characters**: Larger contours were split into two to achieve the required count, ensuring character quality was maintained.

10. **Removal of White Noise**: A final filtering step was applied to remove any residual white spots in segmented images. A minimum area threshold of 1700 pixels was initially set, and if the resulting image was completely black, the threshold was reduced to 100 pixels to retain valid characters.

These techniques collectively enhanced contour detection accuracy, ensuring robust character segmentation. The use of Canny edge detection was particularly beneficial in cases where initial thresholding did not produce clear character edges. However, in scenarios where thresholding performed well, the use of Canny neither improved nor worsened the final results. This phase did not involve training models, focusing solely on optimizing character segmentation from license plate images for better recognition accuracy.

---

### 3.2.3 Recognition

In the initial recognition stage, a single model was used to train on both letters and numbers simultaneously. However, this approach proved inefficient due to the visual similarity between some characters, leading to misclassifications. For instance, letters were often mistaken for numbers and vice versa, significantly reducing the model's accuracy.

To address this issue, the recognition process was divided into two parts: one for letters and another for numbers. This separation optimized the training process for each type of character, improving overall system performance.

#### Letter Recognition
An initial model using Support Vector Machine (SVM) classified letters based on flattened vectors from character images. However, this approach struggled with visually similar characters. A Convolutional Neural Network (CNN) was introduced to extract features, serving as input for the SVM classifier and improving accuracy.

Key elements for letter recognition:
- **CNN Architecture**: Two convolutional layers with 32 and 64 filters, followed by max-pooling and batch normalization.
- **Input Preprocessing**: Images resized to 30x30 pixels and converted to grayscale.
- **SVM Classifier**: A linear kernel was used for efficient letter classification.

#### Number Recognition
Initially, an SVM was used for digit classification, but it faced similar challenges as the letter model. A standalone CNN was implemented, which proved more effective.

Key elements for number recognition:
- **CNN Architecture**: Three convolutional layers with 32 and 64 filters, max-pooling, dropout (0.25), and a dense layer with 128 neurons.
- **Input Preprocessing**: Images resized to 33x47 pixels and converted to grayscale.
- **Output**: A softmax function classified digits from 0-9.

#### Metrics and Evaluation
Both models were evaluated on segmented characters from license plates in the test dataset. Key metrics included:
- **Precision**: Measures the ability to avoid false positives by correctly identifying characters.
- **Recall**: Ensures the detection of all true characters, minimizing false negatives.
- **F1-Score**: Combines precision and recall to evaluate overall performance.
- **Confusion Matrices**: Provided insights into common misclassifications, highlighting areas for improvement.

These metrics revealed strengths and weaknesses in the models, guiding further optimizations to improve recognition accuracy.


### 4. Results

#### 4.1 YOLOv5 Results
The results obtained during the validation of the YOLOv5 model were highly positive. The metrics of precision, recall, and loss showed consistent trends, with training and validation results closely aligned. Key observations include:

- **Precision**: The model achieved a precision value of 0.9 around the 15th epoch, stabilizing at approximately 0.95 by the end of training. This indicates a strong ability to correctly identify license plates without false positives.
- **Recall**: While learning slower than precision, recall reached 0.9 after 20 epochs and continued improving, stabilizing at approximately 0.92. This reflects the model's ability to detect most license plates in the dataset.
- **Loss**:
  - **Box Loss**: Gradually decreased to approximately 0.02, demonstrating improved accuracy in drawing bounding boxes around license plates.
  - **Object Loss**: Reduced to about 0.005, indicating enhanced detection of objects (license plates).

The model showed no signs of overfitting, as training and validation trends were similar. Validation was conducted using images from the Kaggle training dataset, achieving consistent results. When tested on the custom dataset of 47 images, the model successfully localized license plates in all cases.

#### 4.2 Recognition

The following sections present the precision, recall, and F1-Score metrics for letter and number recognition models, as well as a combined model for letters and numbers. 

The models were evaluated on the custom test dataset, with the training datasets producing perfect results. The focus was on assessing performance with unseen data to determine the models' ability to generalize.

##### 4.2.1 Letter Recognition
The SVM-based initial model achieved a precision of 0.89, a recall of 0.93, and an F1-Score of 0.89. Introducing a CNN for feature extraction significantly improved results, with the final model achieving:
- **Precision**: 0.94
- **Recall**: 0.95
- **F1-Score**: 0.94

##### 4.2.2 Number Recognition
The initial SVM model struggled with number recognition, achieving:
- **Precision**: 0.65
- **Recall**: 0.61
- **F1-Score**: 0.56

Replacing the SVM with a CNN resulted in substantial improvements:
- **Precision**: 0.97
- **Recall**: 0.96
- **F1-Score**: 0.96

##### Combined Model for Letters and Numbers
The single model for both letters and numbers performed poorly, with significant misclassifications between the two types. Metrics for the combined model were:
- **Letters**: Precision: 0.74, Recall: 0.69, F1-Score: 0.71
- **Numbers**: Precision: 0.46, Recall: 0.31, F1-Score: 0.35

This result highlights the challenges of combining distinct classes into a single model, emphasizing the need for separate models for letters and numbers.

#### 4.3 Error Analysis and Metrics
Confusion matrices were analyzed for all models, providing insights into common misclassifications:
- **Letter Recognition**: The final model showed minimal errors, primarily with visually similar letters such as "J," "L," or "P," often due to segmentation inaccuracies.
- **Number Recognition**: The CNN model performed well, with occasional errors for the digit "4." The final model significantly outperformed the initial SVM-based approach.
- **Combined Model**: Confusion matrices revealed numerous errors, with frequent misclassifications between letters and numbers. The poor performance validated the decision to use separate models.

#### 4.4 Confidence Intervals for Accuracy
Confidence intervals were computed for the models:
- **Letter Models**:
  - Initial Model: [0.87, 0.97]
  - Final Model: [92.03, 94.73]
  The final model offered a narrower and more reliable interval, ensuring greater confidence in predictions.
  
- **Number Models**:
  - Initial Model: [44.55, 58.20]
  - Final Model: [88.57, 92.23]
  The final model provided a higher minimum accuracy and reduced variability, ensuring improved robustness for real-world applications.

### 5. Conclusions

The results achieved with the YOLOv5 model have been highly positive, with a final precision of approximately 0.95 and a recall of 0.92 during training. This demonstrates high effectiveness in detecting license plates within the test dataset. The letter and number classification models also showed significant improvements, achieving a precision of 0.94 for letters and 0.97 for numbers, reflecting a greater capability to correctly identify these characters.

The use of a convolutional neural network (CNN) for feature extraction in the letter classifier notably enhanced the model's performance. CNNs are particularly effective in capturing complex patterns in images, making them ideal for this task. Additionally, splitting the classification into two independent models, one for letters and one for numbers, proved to be much more efficient than attempting to classify both with a single model. This separation allowed each model to specialize in one type of character, improving the overall system's precision.

A precise segmentation of license plates is essential for effectively recognizing characters. Without accurate segmentation, the models are likely to struggle in correctly identifying letters and numbers, thereby reducing their performance.

Finally, to further enhance the project and broaden its scope to include the recognition of license plates beyond Spanish ones, integrating a continuous learning module could be a valuable addition. Such a module would enable the system to learn and adapt to variations in license plate styles from other countries. This would involve collecting a diversified dataset of international license plates and training the models to recognize these varied patterns, thereby increasing their applicability in a global context.
