# CMakeLists.txt

cmake_minimum_required(VERSION 2.8.12)
project(bindings)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O2")

# if(EXISTS "/Users/jlathrop")    # John's laptop
# set(MOSEK_HOME_DIR "/Users/jlathrop/mosek/10.0/tools/platform/osxaarch64")
# elseif(EXISTS "/home/ben")      # Ben's laptop
# set(MOSEK_HOME_DIR "/home/ben/mosek/mosek/9.3/tools/platform/linux64x86")
# # set(MOSEK_HOME_DIR "/home/ben/mosek/10.1/tools/platform/linux64x86")
# elseif(EXISTS "/home/arcl")
# set(MOSEK_HOME_DIR "/home/arcl/mosek/10.0/tools/platform/linux64x86")
# elseif(EXISTS "/home/pl")      # lambda
# set(MOSEK_HOME_DIR "/home/pl/mosek/mosek/9.3/tools/platform/linux64x86")
# else()
# message("Unknown PC. Unable to locate mosek folder.")
# # /home/ben/anaconda3/envs/dots_env/lib/python3.7/site-packages/mosek
# endif()

# set(MOSEK_SOURCE_DIR ${MOSEK_HOME_DIR}/bin)
# set(MOSEK_INCLUDE_DIR ${MOSEK_HOME_DIR}/h)
# set(ENV_DIR "~/anaconda3/envs/dots_env")
set(ENV_DIR "~/anaconda3/envs/sets")

# add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# find_package(spdlog REQUIRED)
# find_package(fmt REQUIRED)

include_directories( ${EIGEN3_INCLUDE_DIRS} )
# include_directories( ${MOSEK_INCLUDE_DIR} )
include_directories( ${YAML_CPP_INCLUDE_DIRS})
include_directories( "${ENV_DIR}/include")
include_directories( "${ENV_DIR}/include/yaml-cpp" )

# include_directories( ${SPDLOG_INCLUDE_DIRS})
# include_directories( ${FMT_INCLUDE_DIRS})

# find_library(moseklib
#     NAMES "mosek64"
#     HINTS "${MOSEK_SOURCE_DIR}"
#     REQUIRED)
# find_library(fusionlib
#     NAMES "fusion64"
#     HINTS "${MOSEK_SOURCE_DIR}"
#     REQUIRED)
find_library(yaml-cpp
    NAMES "yaml-cpp"
    HINTS "${ENV_DIR}/lib"
    REQUIRED)

# list( APPEND CMAKE_BUILD_RPATH ${MOSEK_SOURCE_DIR} )

pybind11_add_module(bindings bindings.cpp)
set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET "default")
target_link_libraries(bindings PRIVATE ${YAML_CPP_LIBRARIES})

