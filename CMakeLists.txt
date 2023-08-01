cmake_minimum_required(VERSION 2.8)

PROJECT(yjh_deeplearning)

SET(CMAKE_BUILD_TYPE DEBUG)

if(WIN32)
	message("windows platform")
	set(BUILD_SHARED_LIBS "On")
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
	add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)	
else()
	message("linux platform")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fPIC -Wl,--no-as-needed")
endif()


option(USETORCH "use libtorch" OFF)
message("USE_TORCH is ${USETORCH}")
if(USETORCH)
    message("use libtorch")
    add_definitions(-DUSE_LIBTORCH)
endif()

option(USETENSORRT "use tensorrt" OFF)
message("USE_TENSORRT is ${USETENSORRT}")
if(USETENSORRT)
    message("use tensorrt")
    add_definitions(-DUSE_TENSORRT)
endif()

option(USE_ONNXRUNTIME "use onnxruntime" OFF)
message("USE_ONNXRUNTIME is ${USE_ONNXRUNTIME}")
if(USE_ONNXRUNTIME)
    message("use onnxruntime")
    add_definitions(-DUSE_ONNXRUNTIME)
endif()

option(USE_ONNXRUNTIME_CUDA "use onnxruntime cuda, you must set USE_ONNXRUNTIME on first" OFF)
message("USE_ONNXRUNTIME_CUDA is ${USE_ONNXRUNTIME_CUDA}")
if(USE_ONNXRUNTIME_CUDA)
    message("use onnxruntime")
    add_definitions(-DUSE_ONNXRUNTIME_CUDA)
endif()

option(USE_ASCENDCL "use ascendcl" OFF)
message("USE_ASCENDCL is ${USE_ASCENDCL}")
if(USE_ASCENDCL)
    message("use ascendcl")
    add_definitions(-DUSE_ASCENDCL)
endif()

#set(TENSORFLOW_ROOT_DIR /home/vdtap/net_disk/backup/study/tensorflow/tflib)

find_package(OpenCV REQUIRED )
#根据需要进行修改路径
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/yjh
	${PROJECT_SOURCE_DIR}/third_party/clipper
    ${PROJECT_SOURCE_DIR}/app/face_recognition
    ${PROJECT_SOURCE_DIR}/app/defect_recognition
	${PROJECT_SOURCE_DIR}/app/algorithm_c_api
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/preprocess
	
    /usr/local/include/glog 
    
    /usr/local/cuda/include/ 
 
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/Ascend/ascend-toolkit/6.0.RC1/include/
	)




#根据需要修改依赖库名称，openccv的库名称，不同平台，不同版本都不一样 
SET(LINK_LIBRARIES_FILE  yjh_deeplearning ${OpenCV_LIBS}  glog)
if(USETORCH)
    message("use libtorch link")
    SET(LINK_LIBRARIES_FILE  ${LINK_LIBRARIES_FILE} c10 c10_cuda torch_cpu torch_cuda)    
endif()

#注意tensorrt自定义算子库路径，根据需要进行修改
if(USETENSORRT)
    message("use tensorrt link")
    SET(LINK_LIBRARIES_FILE  ${LINK_LIBRARIES_FILE} nvinfer nvinfer_plugin cudart /dongbangfa/code/mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so)    
endif()

if(USE_ONNXRUNTIME)
    message("use onnxruntime link")
    SET(LINK_LIBRARIES_FILE  ${LINK_LIBRARIES_FILE} onnxruntime)    
endif()

if(USE_ASCENDCL)
    message("use ascendcl link")
    SET(LINK_LIBRARIES_FILE  ${LINK_LIBRARIES_FILE} ascendcl)    
endif()

message(${LINK_LIBRARIES_FILE})

#根据需要进行修改路径
LINK_DIRECTORIES(

	/usr/local/cuda/lib64
	/usr/local/lib
	/usr/local/Ascend/ascend-toolkit/6.0.RC1/lib64
	)


AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/source DIR_CORE)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/source/preprocess DIR_SRCS_PREPROCESS)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/app/defect_recognition DIR_SRCS_DEFECT_RECOGNITION)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/app/face_recognition DIR_SRCS_FACE_RECOGNITION)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/app/algorithm_c_api DIR_SRCS_ALGORITHM_C_API)
AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/third_party/clipper DIR_SRCS_CLIPPER)

SET(DIR_SRCS    ${DIR_CORE}
                ${DIR_SRCS_PREPROCESS}
                ${DIR_SRCS_DEFECT_RECOGNITION}
                ${DIR_SRCS_FACE_RECOGNITION}
				${DIR_SRCS_CLIPPER}
				${DIR_SRCS_ALGORITHM_C_API}) 
message(${DIR_SRCS})

SET( LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

ADD_LIBRARY(yjh_deeplearning SHARED ${DIR_SRCS})
TARGET_LINK_LIBRARIES(yjh_deeplearning ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_u2net  ${PROJECT_SOURCE_DIR}/examples/test_u2net.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_u2net ${LINK_LIBRARIES_FILE})


ADD_EXECUTABLE(${PROJECT_NAME}_test_jufang  ${PROJECT_SOURCE_DIR}/examples/test_jufang.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_jufang   ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_yolov5  ${PROJECT_SOURCE_DIR}/examples/test_yolov5.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_yolov5   ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_trt  ${PROJECT_SOURCE_DIR}/examples/test_trt.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_trt   ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_orientedrcnn  ${PROJECT_SOURCE_DIR}/examples/test_oriented.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_orientedrcnn   ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_face_alg  ${PROJECT_SOURCE_DIR}/examples/test_face_alg.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_face_alg   ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_facecpp_alg  ${PROJECT_SOURCE_DIR}/examples/test_facecpp_alg.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_facecpp_alg  ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_face_a200  ${PROJECT_SOURCE_DIR}/examples/test_face_a200.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_face_a200  ${LINK_LIBRARIES_FILE})


ADD_EXECUTABLE(${PROJECT_NAME}_test_recognition_ep  ${PROJECT_SOURCE_DIR}/examples/test_recognition_ep.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_recognition_ep ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_keypoint  ${PROJECT_SOURCE_DIR}/examples/test_keypoint.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_keypoint ${LINK_LIBRARIES_FILE})


ADD_EXECUTABLE(${PROJECT_NAME}_test_keypoint_ep  ${PROJECT_SOURCE_DIR}/examples/test_keypoint_ep.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_keypoint_ep ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_dbnet  ${PROJECT_SOURCE_DIR}/examples/test_dbnet.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_dbnet ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_crnn  ${PROJECT_SOURCE_DIR}/examples/test_crnn.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_crnn ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_ocr  ${PROJECT_SOURCE_DIR}/examples/test_ocr.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_ocr ${LINK_LIBRARIES_FILE})

ADD_EXECUTABLE(${PROJECT_NAME}_test_knob  ${PROJECT_SOURCE_DIR}/examples/test_knob.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME}_test_knob ${LINK_LIBRARIES_FILE})




