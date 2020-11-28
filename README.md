# Master Thesis

References:

https://github.com/xingyizhou/CenterNet

https://bitbucket.org/alberto_pretto/d2co/src/master/

How to use CenterNet with your dataset:

- put your data(annotations and images) in the data folder, the annotations file has to be in COCO format. 
- create a file "my_dataset.py" in CenterNet/src/lib/datasets/dataset/
- modify the following file with your dataset's info:

    -/CenterNet/src/lib/datasets/dataset_factory.py
    -/CenterNet/src/lib/opts.py
    -/CenterNet/src/lib/utils/debugger.py

#train

python main.py ddd --dataset my_dataset --batch_size 16 --master_batch 7 --num_epochs 10 --lr_step 45,60 --gpus 1

The model created by the training is located by default in /exp/ddd/default/

#demo

python demo.py ddd --demo ../data/my_dataset/images/ --load_model ../exp/ddd/default/model_last.pth

How to test CenterNet with the Kitti dataset:

- download the kitti dataset already in COCO format here: https://drive.google.com/drive/folders/11ab9_VLvncWKor2FmJkhdgoorAdDkEt8?usp=sharing
- unzip the content of folder in CenterNet/data/kitti/
- use the previous commands changing the name of the dataset

-------------------------

How to use d2co:

- put your 3D CAD model in /bin/3D_models/
- put your image and your camera calibration file in /bin/test_images/
- run ./test_model -m 3D_models/my_model.stl -c test_images/my_test_camera_calib.yml -i test_images/my_image.jpg --rgb FF0000 --light
- use [j-l-k-i-u-o] and [a-s-q-e-z-w] to move the model
- use b to project the 3d bounding box
- when the model coincide with the object in the image use y to print the info for the annotations in an external file.




