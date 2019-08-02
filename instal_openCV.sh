#============================Install Opencv 3===================================================================
light_Green='\e[1;32m'
NC='\e[0m' # No Color
#font colors:
#Black				0;30		Dark Gray			1;30
#Blue					0;34		Light Blue		1;34
#Green				0;32		Light Green		1;32
#Cyan					0;36		Light Cyan		1;36
#Red					0;31		Light Red			1;31
#Purple				0;35		Light Purple	1;35
#Brown/Orange	0;33		Yellow				1;33
#Light Gray		0;37		White					1;37

echo "${blue}Install Dependence${NC}"
sudo apt -y install build-essential
#sudo apt -y install aptitude
#sudo aptitude -y install libgtk2.0-dev
sudo apt -y install qt-sdk
sudo apt -y install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt -y install python-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev


echo "${blue}Clone to GitHub${NC}"
sudo rm  -r ~/OpenCV3
mkdir ~/OpenCV3
cd ~/OpenCV3
git clone https://github.com/Itseez/opencv.git
git checkout 3.4.0

echo "${blue}Run CMake${NC}"
cd opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_GTK=ON -D WITH_OPENGL=ON ..

echo "${blue}Copile and install${NC}"
make -j4 # Numero de processadores na maquina
sudo make install

echo "${light_Green}Finalizado${NC}"
sudo rm  -r ~/OpenCV3
sleep 1
