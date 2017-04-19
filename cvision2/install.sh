# Install required ubuntu packages.
sudo apt-get install build-essential cmake git pkg-config
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
sudo apt-get install libgtk2.0-dev
sudo apt-get install libatlas-base-dev gfortran

# Create an opencv dir and download the latest version.
mkdir opencv_src
cd opencv_src/
git clone https://github.com/Itseez/opencv.git
cd opencv/

# Cmake it.
mkdir release
cd release/
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make

# Install it in env/local/lib
sudo make install

# Link it to site-packages
cd env/lib/python2.7/site-packages/
ln -s /usr/local/lib/python2.7/dist-packages/cv2.so .

# Finally create the virutalenv, and install numpy
virtualenv env
source env/bin/activate
pip install numpy
pip install Pillow

