cmake_minimum_required(VERSION 3.18...3.31)
project(lighthit_g4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Geant4 built with static libraries (BUILD_STATIC_LIBS=ON,
# GEANT4_BUILD_TLS_MODEL=global-dynamic, position-independent code); point
# Geant4_DIR or CMAKE_PREFIX_PATH at its installation.
find_package(Geant4 11.4 REQUIRED COMPONENTS static)

nanobind_add_module(_core STABLE_ABI NB_STATIC cpp/core.cpp cpp/external_decayer.cpp)
target_include_directories(_core PRIVATE cpp)
target_link_libraries(_core PRIVATE ${Geant4_LIBRARIES})

install(TARGETS _core LIBRARY DESTINATION lighthit_g4)
