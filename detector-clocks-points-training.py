import dlib

options = dlib.shape_predictor_training_options()
dlib.train_shape_predictor("assets/treinamento_relogios_pontos.xml", "assets/detector_relogios_pontos.dat", options)



