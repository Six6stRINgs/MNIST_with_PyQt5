# MNIST System

## Description
An assignment for my school coures software engineering.

This is a simple system that allows users to draw a digit and then classify it using a neural network. 
The system is built using Python and PyQt5. 
The neural network is built using PyTorch and is trained on the MNIST dataset.

Database is used only to store the user's information.

## Environment

- Python
- PyQt5
- mysql_connector_repackaged
- netron
- numpy
- onnxruntime
- Pillow
- PyQt5_sip
- torch
- torchvision

## Installation

Install the required packages by running the following command:

```bash
pip install -r requirements.txt
```
## Run

```Activate the software
python python main.py 
        --host {your_db_host} 
        --user {your_db_user} 
        --password {your_db_password} 
        --database {your_db_name} 
        --login_without_info {True/False}
```

## Software Snapshots
Login Dialog:

<p align="center">
  <img src="./sys_picture/3.png" width="200" alt="">
</p>

System MainWindow:
![image](./sys_picture/1.png)
![image](./sys_picture/2.png)
![image](./sys_picture/4.png)
![image](./sys_picture/5.png)
