# brain-tumor-pretrained
# brain-tumor-fs
<h1>Brain tumor detection using VGG16 and smaller dataset</h1>

<p>Number of images:<p>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/num_images.png" alt="Number of images" width="auto" height="300"> 

<h2>Model architecture</h2>
<p>Img target_size=[214,214]</p>
<p>Sequential model with layers:</p>
<ul>
  <li>VGG16 base_model with weights="imagenet"</li>
  <li>Dropout(0.3)</li></li>
  <li>Flatten()</li>
  <li>Dropout(0.5)</li>
  <li>Dense(1, activation='sigmoid')</li>
 </ul>
 
 <p>

</p>
<br>
<h3>Training vs Validation accuracy</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/accuracy.png" alt="" width="auto" height="300"> 
<h3>Training vs Validation loss</h3>
<img src="" alt="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/loss_function.png" width="auto" height="300"> 
<h3>confusion matrix for validation set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_val_data.png" alt="" width="auto" height="300"> 
<h3>confusion matrix for test set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_test_data.png" alt="" width="auto" height="300"> 

<hr><hr>

<h1>Brain tumor v2: using bigger dataset</h1>

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
<img src="" alt="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/loss_function_bt.png" width="auto" height="300"> 
<h3>confusion matrix for validation set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_val_data_bt.png" alt="" width="auto" height="300"> 
<h3>confusion matrix for test set</h3>
<img src="https://raw.githubusercontent.com/rastislavkopal/brain-tumor-pretrained/master/graphs/cm_test_data_bt.png" alt="" width="auto" height="300"> 
