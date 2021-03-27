Begin
====================================================================================================
How to use:
----------------------------------------------------------------------------------------------------
1.Every images of same type must have the same tag and all put into a folder witch named by the tag.Run the script Launcher.py in the project folder.<br>
2.To import the images and build the dataset for tranning,create folders,rename them with the tags,and put images correctly into them.Then put the folders into the "raw" folder in the project folder.You can simply import single folder by click the "Import" button on the user interface.<br>
3.Click the "Process data" to load and process the images to the same size so the model can use them for training.If you import data by click the "Import" button,the script will process the images automatically.<br>
4.To start trainning,just click the "Train" button.Make sure the cuda toolkit and cuDNN installed on your device.When its done,it will show you the accuracy and ask you to save the model or not.The model will save as "date+time.h5" in the "saved_model" folder.<br>
5.Now you can start recognize.Click the "Recognize" button and choose a folder.The script will recognize all the images in the folder whitch you have chose.<br>

Use in console:
----------------------------------------------------------------------------------------------------
You can just run the Core.py in the console,by using "python ./Core.py [operation]" in the project folder.<br>
To process data:python ./Core 0<br>
To recognize:python ./Core 1 [Your path]<br>
To start trainning:python ./Core 2<br>

This project should run in tensorflow2.You can simply install the libraries this project needs by running "envinstall.sh".
