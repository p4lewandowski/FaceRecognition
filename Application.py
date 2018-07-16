from FaceRecognition import FaceRecognition
from ImagePreprocessing import image_selection

# Select images where face can be detected
# image_selection()
# Create object containing functions to calculate eigenfaces
fr = FaceRecognition()
# Import images
fr.get_images()
# Find eigenfaces
fr.get_eigenfaces()
# Plot results
fr.show_me_things()
# Find face closest to the one specified
fr.find_me_face('new_face.jpg')