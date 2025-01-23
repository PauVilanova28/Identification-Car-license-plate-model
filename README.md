# Identification Car License Plate Model

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
