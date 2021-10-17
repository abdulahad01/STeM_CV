import tensorflow as tf
model = tf.keras.models.load_model( "/home/pi/Downloads/STeM_CV/detector1.model" )
converter = tf.lite.TFLiteConverter.from_keras_model( model )
open( 'model.tflite' , 'wb' ).write( converter.convert() )
