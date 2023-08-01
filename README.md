#dlalg是一个集成了多个推理引擎的c++推理框架sdk，用户利用此框架，免去花费很多精力去写推理代码的问题。此框架支持多个级联模型，支持opencv_dnn,tensorrt,libtorch,onnxruntime,华为acl推理框架。此框架基本的依赖是opencv和glog。

#目录结构
app:算法业务应用源码目录
src:核心源码目录
examples:推理例子目录
config:配置文件目录
model:模型文件目录
lib:sdk

#编译依赖，用户需要自行编译opencv，glog。并在CmakeLits.txt文件配置路径。框架默认支持opencv_dnn。如果需要支持tensorrt,libtorch,onnxruntime，用户需要打开CMakeLists.txt中的相关选项。
编译步骤：
mkdir build
cd build
make -j4

#在lib目录中生成libyjh_deeplearning.so文件，用户将此文件和dl_algorithm.h文件、相关配置文件和模型文件。这些文件就可以形成算法的输出，其他业务可以直接调用。调用的例子详见example中的示例代码。





