


echo -e "${blue}Install TensorFlow${NC}";
sudo pip install tensorflow-GPU      # for Python 2.7
#if sudo pip install tensorflow-gpu; then  # Python 2.7;  GPU support
#	echo -e "${red}TensorFlow succesfully installed";
#fi
#else
#	sudo pip  install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl
#    Python 2.7
#fi
echo -e"${red}Tensorflow succesfully installed"

echo -e "${blue}Install Protoc_3.5.1${NC}";
mkdir ~/protoc_3.5.1
cd ~/protoc_3.5.1
if wget https://github.com/google/protobuf/releases/download/v3.5.1/protoc-3.5.1-linux-x86_64.zip && sudo chmod 775 protoc-3.5.1-linux-x86_64.zip && unzip protoc-3.5.1-linux-x86_64.zip; then
	echo -e"${red}Protocol buffers succesfully installed"
fi

#Or run 'sudo pip install protobuf==3.5.1'


echo -e "${blue}Clone Models${NC}";
cd ~/
git clone https://github.com/tensorflow/models.git

cd models/
git checkout 3a05570f

echo -e "${blue}Install Models${NC}";
# From ~/models/research/
cd ~/models/research/
	~/protoc_3.5.1/bin/protoc object_detection/protos/*.proto --python_out=.
echo "export PYTHONPATH=~/models/research/slim:~/models/research/object_detection/utils:~/models/research:\$PYTHONPATH" >> ~/.bashrc;

#-------------------------------------------


