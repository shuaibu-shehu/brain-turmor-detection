# Brain-Tumor-Detection-Using-Deep-Learning-MRI-Images-Detection-Using-Computer-Vision

<h2>What is a brain tumor?</h2>
A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems. Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.

<h2>The importance of the subject</h2>
Early detection and classification of brain tumors is an important research domain in the field of medical imaging and accordingly helps in selecting the most convenient treatment method to save patients life therefore

<h2>Methods</h2>
The application of deep learning approaches in context to improve health diagnosis is providing impactful solutions. According to the World Health Organization (WHO), proper brain tumor diagnosis involves detection, brain tumor location identification, and classification of the tumor on the basis of malignancy, grade, and type. This experimental work in the diagnosis of brain tumors using Magnetic Resonance Imaging (MRI) involves detecting the tumor, classifying the tumor in terms of grade, type, and identification of tumor location. This method has experimented in terms of utilizing one model for classifying brain MRI on different classification tasks rather than an individual model for each classification task. The Convolutional Neural Network (CNN) based multi-task classification is equipped for the classification and detection of tumors. The identification of brain tumor location is also done using a CNN-based model by segmenting the brain tumor.

![mri](https://github.com/user-attachments/assets/c372c4c6-dab2-4835-b1d3-a7b24217cac9)

<h2>MODEL:</h2>
I AM USING VGG16 FOR TRANSFER LEARNING.
The model is built on top of VGG16, which is a pre-trained convolutional neural network (CNN) for image classification.

First, the VGG16 model is loaded with input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet'. The input shape is set to match the size of the images in the dataset, which is 128x128 pixels. The include_top parameter is set to False, which means that the final fully-connected layers of VGG16 that perform the classification will not be included. The weights parameter is set to 'imagenet' which means that the model will be pre-trained with a dataset of 1.4 million images called imagenet

Next, the for layer in base_model.layers: loop is used to set all layers of the base_model (VGG16) to non-trainable, so that the weights of these layers will not be updated during training.

Then, the last three layers of the VGG16 model are set to trainable by using base_model.layers[-2].trainable = True,base_model.layers[-3].trainable = True and base_model.layers[-4].trainable = True

After that, a Sequential model is created and the VGG16 model is added to it with model.add(base_model).

Next, a Flatten layer is added to the model with model.add(Flatten()) which reshapes the output of the VGG16 model from a 3D tensor to a 1D tensor, so that it can be processed by the next layers of the model.

Then, a Dropout layer is added with model.add(Dropout(0.3)) which is used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.

After that, a dense layer is added with 128 neurons and relu activation function is added with model.add(Dense(128, activation='relu')).

Next, another Dropout layer is added with model.add(Dropout(0.2))

Finally, the output dense layer is added with number of neurons equal to the number of unique labels and 'softmax' activation function is added with model.add(Dense(len(unique_labels), activation='softmax')). The 'softmax' activation function is used to give a probability distribution over the possible classes.

<div align="center" style="
  display: flex;
  flex-direction: column;
  gap: 50px; /* Adjust for smaller gaps */
  align-items: center; /* Center aligns images horizontally */
  padding: 20px; /* Adds padding around the container */
  background-color: #f9f9f9; /* Light gray background */
  border-radius: 10px; /* Rounded corners */
">

  <img style="width:80%; max-width: 600px;" src="https://github.com/user-attachments/assets/1f4e4023-5770-4d35-80f6-2760fbf752c7" alt="Picture2">
  <img style="width:80%; max-width: 600px;" src="https://github.com/user-attachments/assets/a7185a5f-2578-4d96-bc01-604cd9a89b80" alt="Picture1">
  <img style="width:80%; max-width: 600px;" src="https://github.com/user-attachments/assets/30cb5c58-e3fd-4b8b-8421-748d628510c9" alt="Picture3">
  <img style="width:80%; max-width: 600px;" src="https://github.com/user-attachments/assets/fa7feac1-fd5b-4709-af43-6550bb173a03" alt="Picture3">
  
</div>
