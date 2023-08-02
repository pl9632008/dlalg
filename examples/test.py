#coding=utf-8
import ctypes
import sys

# 加载共享库
yjhalg_lib = ctypes.CDLL('../lib/libyjh_deeplearning.so')

# 定义C++函数签名
InitXuanniuModel = yjhalg_lib.InitXuanniuModel
InitXuanniuModel.argtypes = (ctypes.c_char_p, ctypes.c_int)
InitXuanniuModel.restype = ctypes.c_void_p

config_path = sys.argv[1]
img_path = sys.argv[2]

engine = InitXuanniuModel(ctypes.c_char_p(config_path.encode('utf-8')),0)

XuanniuStatusRecognition = yjhalg_lib.XuanniuStatusRecognition
XuanniuStatusRecognition.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
XuanniuStatusRecognition.restype = ctypes.c_char_p

res_str = XuanniuStatusRecognition(ctypes.c_void_p(engine),ctypes.c_char_p(img_path))

print(res_str)
