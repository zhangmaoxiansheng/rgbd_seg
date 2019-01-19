import os
os.system("python seg_rgbd_flexible.py -i ./pc12 -th 2700 -s 430 -e 1473 -o ./pc12/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq1_12 -th 2800 -s 430 -e 1451 -o ./zmq1_12/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq2_12 -th 2900 -s 430 -e 1451 -o ./zmq2_12/nxp")
os.system("python seg_rgbd_flexible.py -i ./pc4 -th 2700 -s 743 -e 1900 -o ./pc4/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq1_4 -th 2700 -s 600 -e 1900 -o ./zmq1_4/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq2_4 -th 2900 -s 750 -e 1900 -o ./zmq2_4/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq1_5 -th 2900 -s 290 -e 1900 -o ./zmq1_5/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq2_5 -th 3000 -s 300 -e 1900 -o ./zmq2_5/nxp")

