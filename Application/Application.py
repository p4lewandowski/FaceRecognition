from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces
from FaceRecognition_ImagePreprocessing import image_selection
import os

# efr = EigenfaceRecognitionNewfaces(filepath=os.path.join(os.getcwd(), '..', 'Data',
#                                                          'Database\\212images-36people.p'))
fr = FaceRecognitionEigenfaces()
fr.get_images()
fr.get_eigenfaces()
fr.save_to_file()
# fr.show_me_things()
efr = EigenfaceRecognitionNewfaces(data=fr)

# efr.add_person()
# efr.add__new_face('a.pgm')
# efr.face_data.get_eigenfaces()
# efr.face_data.show_me_things()
