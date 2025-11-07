cmake_minimum_required(VERSION 3.22.1)
project(bytetrack_jni_project)

# ========== Option: 选择是否编译 hybrid tracker 的 JNI 桥 ==========
# 如果你想使用旧的 ByteTrack JNI，把 ON 改为 OFF；也可以在 gradle 中传 -DBUILD_HYBRID=OFF
option(BUILD_HYBRID "Build using HybridTracker JNI (if ON) otherwise build ByteTrackJNI" ON)
message(STATUS "BUILD_HYBRID = ${BUILD_HYBRID}")

# ========== OpenCV / Eigen: 允许从 Gradle 传入 -DOpenCV_DIR/-DEIGEN3_INCLUDE_DIR ==========
if(DEFINED OpenCV_DIR)
    message(STATUS "OpenCV_DIR provided: ${OpenCV_DIR}")
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
endif()

if(NOT OpenCV_FOUND)
    find_package(OpenCV REQUIRED)
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/eigen" CACHE PATH "Eigen include directory")
endif()
message(STATUS "Eigen include path: ${EIGEN3_INCLUDE_DIR}")

# ========== 明确列出你当前目录下存在的源文件（按你目录列的名单） ==========
set(SRC_FILES
        BYTETracker.cpp
        STrack.cpp
        kalmanFilter.cpp
        lapjv.cpp
        utils.cpp
        # 下面两个 JNI 桥接按 BUILD_HYBRID 决定是否加入
)

# 如果存在 Lightweight/Hybrid 源，加入它们（一般存在）
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/LightweightTracker.cpp")
    list(APPEND SRC_FILES LightweightTracker.cpp)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/HybridTracker.cpp")
    list(APPEND SRC_FILES HybridTracker.cpp)
endif()

# 根据 BUILD_HYBRID 选择 JNI 桥接文件（避免同时包含两个 JNI 导出冲突）
if(BUILD_HYBRID)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/HybridTrackerJNI.cpp")
        list(APPEND SRC_FILES HybridTrackerJNI.cpp)
        message(STATUS "Using HybridTrackerJNI.cpp")
    else()
        message(FATAL_ERROR "BUILD_HYBRID=ON but HybridTrackerJNI.cpp not found in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
else()
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ByteTrackJNI.cpp")
        list(APPEND SRC_FILES ByteTrackJNI.cpp)
        message(STATUS "Using ByteTrackJNI.cpp")
    else()
        message(FATAL_ERROR "BUILD_HYBRID=OFF but ByteTrackJNI.cpp not found in ${CMAKE_CURRENT_SOURCE_DIR}")
    endif()
endif()

# 最后把其它必要文件确认加入（可选警告）
foreach(f ${SRC_FILES})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
        message(WARNING "Expected source file missing: ${f}")
    endif()
endforeach()

# 如果没有任何源，显式终止并给诊断提示
if(NOT SRC_FILES)
    message(FATAL_ERROR "No source files selected for bytetrack_jni. Please check app/src/main/cpp/")
endif()

# 添加库
add_library(bytetrack_jni SHARED ${SRC_FILES})

# include dirs
target_include_directories(bytetrack_jni PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

# 链接 Android 日志 + OpenCV
find_library(log-lib log)
target_link_libraries(bytetrack_jni
        ${log-lib}
        ${OpenCV_LIBS}
)

# 可选：如果需要更严格的链接检查，可以启用下面一行（有时会导致交叉编译问题）
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
