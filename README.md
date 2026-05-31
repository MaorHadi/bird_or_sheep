# arabOrJew_machineLearning
to activate:

(inside bash) => chmod +x setup.sh

(inside bash) => ./setup.sh

(inside bash) => py scripts/bird_or_sheep.py - this should create the model.

(inside bash) => py scripts/activate_model
//the activate model file will take the created model and the image specified at the end of the file, examine the model and release an output


dataset example:

dataset
--bird
----bird_image_1.jpg
----bird_image_2.jpg
----bird_image_3.jpg
----birdy_blalalala.jpg
----bird_image_5.jpg
----bird_image_67.jpg
--sheep
----sheep_image_1.jpg
----sheep_image_2.jpg
----sheep_image_3.jpg
----sheepy_blalalala.jpg
----sheep_image_5.jpg
----sheep_image_67.jpg


correct output example:
This image is most likely a X with Y% confidence.