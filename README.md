# Depth_Guidance_dehazing

The code of "Image Dehazing via Self-supervised Depth Guidance"
If you prefer pip, install following versions:

timm==0.3.2
torch==1.7.1
torchvision==0.8.2
## Test Code:


To run the evaluation for specific test datasets, run the following commands:
# Testing         
    |   |   ├── <dataset_name>          
    |   |   |   ├── lowq         # hazy images 
    |   |   |   └── clear           # ground truth images
Modify the test dataset path in the test.py file  ,val_data_dir='your path'


python test.py -exp_name trained_model

You can download the trained model for the Roadmap dataset via this link."https://pan.baidu.com/s/1uOKpC315aal6rVJd-o6XxA 
password：re6k"


You can download the trained model for the RESIDEK-6K dataset via this link."https://pan.baidu.com/s/1QGILBVl3a1BVtIH42j8aXw 
password:re6k"



