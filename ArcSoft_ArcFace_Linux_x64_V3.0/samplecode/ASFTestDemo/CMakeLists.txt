# cmake needs this line
cmake_minimum_required(VERSION 3.1)
project(AFCFaceDemo_Linux)				#指定项目的名称，一般和项目的文件夹名称对应
find_package(OpenCV 3 REQUIRED)
set(CMAKE_CXX_STANDARD 11)				#设置c++标准
include_directories(${OpenCV})
#指定头文件目录
include_directories(./)	
include_directories(./inc)
include_directories(${OpenCV_INCLUDE_DIRS}) #加这一句
#指定静态和动态文件目录
link_directories(./linux_so)

#该指令的主要作用就是将指定的源文件生成链接文件，然后添加到工程中去。
#arcsoft_face生成的库文件名		[STATIC | SHARED | MODULE] 库类型
add_executable(arcsoft_face_engine_test 
        ./inc/amcomdef.h
        ./inc/arcsoft_face_sdk.h
		./inc/asvloffscreen.h
		./inc/merror.h
		./samplecode.cpp
		)

#-fPIC	在给定的作用域内设置一个命名的属性
set_property(TARGET arcsoft_face_engine_test PROPERTY POSITION_INDEPENDENT_CODE ON)

# Declare the executable target built from your sources
add_executable(AFCFaceDemo_Linux samplecode.cpp)
#add_executable(AFCFaceDemo_Linux test.cpp)
# Link your application with OpenCV libraries
target_link_libraries(arcsoft_face_engine_test  ${OpenCV_LIBS})
#链接库文件
target_link_libraries(arcsoft_face_engine_test	
		arcsoft_face
		arcsoft_face_engine

        )
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    versi on: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")


