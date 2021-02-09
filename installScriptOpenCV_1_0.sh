#!/bin/bash

# Script to get OpenCV running on a Raspberry Pi 4(00)+
# J.A. Korten, Feb 9. 2021
# Based on: https://pimylifeup.com/raspberry-pi-opencv/

echo
echo "Install script 1/2 for OpenCV on Raspberry Pi 4(00)+"
echo "Version 1.0 - Feb 2021"
echo "Johan Korten - HAN University of Applied Sciences"
echo "School of Engineering and Automotive"
echo ""
echo ""
while true; do
    read -p "Do you really wish to install OpenCV? [Y/N] " yn
    case $yn in
        [Yy]* ) make install; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done

start=$(date +%s.%N)

echo "====================================================="
echo " Step 1.1. Updating system"
echo "====================================================="
echo
sudo apt update -y

echo "====================================================="
echo " Step 1.2. Upgrading system"
echo "====================================================="
echo
sudo apt upgrade -y

echo "====================================================="
echo " Step 1.3. Installing cmake and other essential tools"
echo "====================================================="
echo
sudo apt install cmake build-essential pkg-config git -y

echo "====================================================="
echo " Step 1.4. Installing image format processing tools"
echo "====================================================="
echo
sudo apt install libjpeg-dev libtiff-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev -y

echo "====================================================="
echo " Step 1.5. Installing video format processing tools"
echo "====================================================="
echo
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev -y

echo "====================================================="
echo " Step 1.6. Installing QT interface tools for OpenCV"
echo "====================================================="
echo
sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 -y

echo "====================================================="
echo " Step 1.7. Performance Tools for OpenCV"
echo "====================================================="
echo
sudo apt install libatlas-base-dev liblapacke-dev gfortran -y

echo "====================================================="
echo " Step 1.8. Further with HDF5"
echo "====================================================="
echo
sudo apt install libhdf5-dev libhdf5-103 -y

echo "====================================================="
echo " Step 1.9. The Python tools/libraries"
echo "====================================================="
echo
sudo apt install python3-dev python3-pip python3-numpy -y

echo "====================================================="
echo " Step 2.1. Preparing for compilation,"
echo "           making swap file larger (100Mb -> 2Gb)"
echo "====================================================="

sudo mv /etc/dphys-swapfile /etc/dphys-swapfile_
echo " Old swap file config was renamed to /etc/dphys-swapfile_"

cat > /etc/dphys-swapfile <<EOF
# This swap file config was generated by the OpenCV install script.
CONF_SWAPSIZE=2048
EOF
echo " Temporary 2Gb swap file configuration was initiated..."

echo " Restaring swap file service..."
sudo systemctl restart dphys-swapfile

echo "====================================================="
echo " Step 2.2. Downloading OpenCV sources from Git"
echo "           Cloning git repos."
echo "           This might take some time..."
echo "====================================================="

cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

echo "====================================================="
echo " Step 3.1. Preparing OpenCV for compilation: "
echo "  making appropriate folder... "
echo "====================================================="

mkdir ~/opencv/build
cd ~/opencv/build

echo "====================================================="
echo " Step 3.2. Preparing OpenCV for compilation: "
echo "  creating makefile using cmake "
echo "====================================================="

cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
    -D ENABLE_NEON=ON \
    -D ENABLE_VFPV3=ON \
    -D BUILD_TESTS=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D CMAKE_SHARED_LINKER_FLAGS=-latomic \
    -D BUILD_EXAMPLES=OFF ..

echo "====================================================="
echo " Step 3.3. Going to compile OpenCV, "
echo "  so get yourself some coffee/tea/hot cocoa..."
echo "  It will take quite some time (approx. one hour)..."
echo "====================================================="

make -j$(nproc)

echo "====================================================="
echo " Step 3.4. Installing OpenCV itself "
echo "  phew, almost ready to run..."
echo "====================================================="

sudo make install

echo "====================================================="
echo " Step 4.1. Finishing the installation... "
echo "====================================================="

sudo ldconfig

echo "====================================================="
echo " Step 4.1. Finishing the installation... "
echo "====================================================="

sudo mv /etc/dphys-swapfile /etc/dphys-swapfile__
sudo mv /etc/dphys-swapfile_ /etc/dphys-swapfile
echo " Swap file was set from 2Gb to 100Mb again in /etc/dphys-swapfile"

echo "====================================================="
echo " Step 4.2. Resetting swap file process... "
echo "====================================================="

sudo systemctl restart dphys-swapfile
echo "Done."

echo "====================================================="
echo " Step 4.3. Finished OpenCV install... "
echo "====================================================="
echo

duration=$(echo "$(date +%s.%N) - $start" | bc)
execution_time=`printf "%.2f seconds" $duration`

echo " It took your Pi $execution_time seconds to run this script :)"
echo " We have finished the installation of OpenCV."
echo

echo "====================================================="
echo " Step 5. Installing pip cv2 (last step)"
echo "====================================================="
echo

pip3 install opencv-python

echo
echo " You can now test Python 3 and OpenCV using following lines:"
echo " import cv2"
echo " cv2.__version__"
echo " This should give the version number (e.g. 4.5.1)"
