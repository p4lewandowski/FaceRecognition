from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces
from FaceRecognition_ImagePreprocessing import image_selection

# Select images where face can be detected
# image_selection()
# Create object containing functions to calculate eigenfaces
fr = FaceRecognitionEigenfaces()
# Import images
fr.get_images()
# Find eigenfaces
fr.get_eigenfaces()
# Plot results
fr.show_me_things()
fr.save_to_file()

# efr = EigenfaceRecognitionNewfaces()
# # Find face closest to the one specified
# efr.find_me_face('new_faces_notindetected/new_face.jpg')
# efr.find_me_face_knn('new_faces_notindetected/new_face.jpg')

