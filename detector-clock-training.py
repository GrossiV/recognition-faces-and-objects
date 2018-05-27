import dlib

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5

dlib.train_simple_object_detector("assets/treinamento_relogios.xml", "assets/detector_relogios.svm", options)
