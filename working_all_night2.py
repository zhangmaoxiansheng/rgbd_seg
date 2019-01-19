import os
os.system("python seg_rgbd_flexible.py -i ./pc5 -th 2700 -s 860 -e 1900 -o ./pc5/nxp")
os.system("python seg_rgbd_flexible.py -i ./pc6 -th 2700 -s 500 -e 1900 -o ./pc6/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq1_6 -th 2900 -s 514 -e 1900 -o ./zmq1_6/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq2_6 -th 2900 -s 543 -e 1900 -o ./zmq2_6/nxp")
os.system("python seg_rgbd_flexible.py -i ./pc7 -th 2700 -s 300 -e 1880 -o ./pc7/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq1_7 -th 3000 -s 300 -e 1900 -o ./zmq1_7/nxp")
os.system("python seg_rgbd_flexible.py -i ./zmq2_7 -th 3300 -s 300 -e 1900 -o ./zmq2_7/nxp")

