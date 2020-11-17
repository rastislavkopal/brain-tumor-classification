# brain-tumor-pretrained
<h2>Brain tumor detection using VGG16 and smaller dataset</h2>

<p>Number of images:<p>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/num_images.png" alt="Number of images" width="auto" height="300"> 

<h2>Model architecture</h2>
<p>Img target_size=[128,128]</p>
<p>Sequential model with layers:</p>
<ul>
  <li>VGG16 base_model with weights="imagenet"</li>
  <li>Dropout(0.3)</li></li>
  <li>Flatten()</li>
  <li>Dropout(0.5)</li>
  <li>Dense(1, activation='sigmoid')</li>
 </ul>
 
 <p>
Epoch 42/120
50/50 [==============================] - 8s 153ms/step - loss: 0.4550 - accuracy: 0.7721 - val_loss: 0.0277 - val_accuracy: 0.9742
</p>
<br>
<h3>Training vs Validation accuracy</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/accuracy.png" alt="" width="auto" height="300"> 
<h3>Training vs Validation loss</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/loss_function.png" alt="" width="auto" height="300"> 
<h3>confusion matrix for validation set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_val_data.png" alt="" width="auto" height="300"> 
<h3>confusion matrix for test set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_test_data.png" alt="" width="auto" height="300"> 

<b>loss at test dataset: 0.42<br>
accuracy at test dataset: 0.85 </b>

<hr><hr>

<h2>Brain tumor v2: using bigger dataset</h2>

<p>Number of images:<p>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/num_images_bt.png" 

<h2>Model architecture</h2>
<p>Img target_size=[128,128]<br>
  Using same model
</p>
 
 <p>
Epoch 10/120
50/50 [==============================] - 8s 152ms/step - loss: 0.4591 - accuracy: 0.8233 - val_loss: 0.3263 - val_accuracy: 0.8111
</p>
<br>
<h3>Training vs Validation accuracy</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/accuracy_bt.png" alt="" width="auto" height="300"> 
<h3>Training vs Validation loss</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/loss_function_bt.png" alt="" width="auto" height="300"> 
<h3>confusion matrix for validation set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_val_data_bt.png" alt="" width="auto" height="300"> 
<h3>confusion matrix for test set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_test_data_bt.png" alt="" width="auto" height="300"> 

<b>loss at test dataset: 0.46<br>
accuracy at test dataset: 0.85</b>
<hr>
<h3>After improvements</h3>
<p>
poch 20/20
50/50 [==============================] - 16s 327ms/step - loss: 0.8566 - accuracy: 0.8908 - val_loss: 1.2597 - val_accuracy: 0.9281<br>
Time needed for training:  331.61661171913147 s.<br>
Val Accuracy = 0.92<br>
Test Accuracy = 0.96
</p>
<h3>After cropped images (contour)</h3>
Epoch 20/20
50/50 [==============================] - 17s 335ms/step - loss: 0.9749 - accuracy: 0.8982 - val_loss: 2.7541 - val_accuracy: 0.9144<br>
Time needed for training:  336.28277111053467 s.<br>
Val Accuracy = 0.92<br>
Test Accuracy = 0.95<br>
