# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/iaslab/TesiRC/d2co

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/iaslab/TesiRC/d2co/build

# Include any dependencies generated for this target.
include CMakeFiles/obj_localization.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/obj_localization.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/obj_localization.dir/flags.make

CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o: CMakeFiles/obj_localization.dir/flags.make
CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o: ../apps/obj_localization.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/iaslab/TesiRC/d2co/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o -c /home/iaslab/TesiRC/d2co/apps/obj_localization.cpp

CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/iaslab/TesiRC/d2co/apps/obj_localization.cpp > CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.i

CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/iaslab/TesiRC/d2co/apps/obj_localization.cpp -o CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.s

CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.requires:

.PHONY : CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.requires

CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.provides: CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.requires
	$(MAKE) -f CMakeFiles/obj_localization.dir/build.make CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.provides.build
.PHONY : CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.provides

CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.provides.build: CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o


CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o: CMakeFiles/obj_localization.dir/flags.make
CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o: ../apps/apps_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/iaslab/TesiRC/d2co/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o -c /home/iaslab/TesiRC/d2co/apps/apps_utils.cpp

CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/iaslab/TesiRC/d2co/apps/apps_utils.cpp > CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.i

CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/iaslab/TesiRC/d2co/apps/apps_utils.cpp -o CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.s

CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.requires:

.PHONY : CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.requires

CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.provides: CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/obj_localization.dir/build.make CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.provides.build
.PHONY : CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.provides

CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.provides.build: CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o


# Object files for target obj_localization
obj_localization_OBJECTS = \
"CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o" \
"CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o"

# External object files for target obj_localization
obj_localization_EXTERNAL_OBJECTS =

../bin/obj_localization: CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o
../bin/obj_localization: CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o
../bin/obj_localization: CMakeFiles/obj_localization.dir/build.make
../bin/obj_localization: ../lib/libd2co.a
../bin/obj_localization: ../cv_ext/lib/libcv_ext.a
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libuuid.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.5.2
../bin/obj_localization: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libdime.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkDomainsChemistry-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneric-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersHyperTree-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelFlowPaths-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersFlowPaths-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelGeometry-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelImaging-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelMPI-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallelStatistics-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersProgrammable-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersPython-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/libvtkWrappingTools-6.3.a
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersReebGraph-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersSMP-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersSelection-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersTexture-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersVerdict-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkverdict-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOAMR-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOEnSight-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOExport-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingGL2PS-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingContextOpenGL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOFFMPEG-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOMovie-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOGDAL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOGeoJSON-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOImport-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOInfovis-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOMINC-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOMPIImage-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOMPIParallel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOParallel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIONetCDF-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOMySQL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOODBC-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOParallelExodus-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOExodus-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkexoIIc-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOParallelLSDyna-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOLSDyna-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOParallelNetCDF-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOParallelXML-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOPostgreSQL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOVPIC-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkVPIC-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOVideo-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOXdmf2-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkxdmf2-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingMath-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingMorphological-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingStatistics-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingStencil-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkLocalExample-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI4Py-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingExternal-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingImage-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingMatplotlib-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkWrappingPython27Core-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkPythonInterpreter-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersParallel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingParallelLIC-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkParallelMPI-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingLIC-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeAMR-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersAMR-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkParallelCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolumeOpenGL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libSM.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libICE.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libX11.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libXext.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libXt.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkTestingGenericBridge-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkTestingIOSQL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOSQL-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkTestingRendering-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkViewsGeovis-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkViewsInfovis-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersImaging-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingLabel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkGeovisCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOXML-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOXMLParser-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkInfovisLayout-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkInfovisBoostGraphAlgorithms-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkInfovisCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersHybrid-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingGeneral-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingHybrid-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOImage-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkIOCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkmetaio-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkftgl-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingColor-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingVolume-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersStatistics-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingFourier-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkalglib-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonSystem-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtksys-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkWrappingJava-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3.0
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_common.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
../bin/obj_localization: /usr/lib/libOpenNI.so
../bin/obj_localization: /usr/lib/libOpenNI2.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libexpat.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libgl2ps.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_io.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_search.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_features.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_people.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpthread.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_system.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_thread.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_serialization.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libboost_regex.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_common.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
../bin/obj_localization: /usr/lib/libOpenNI.so
../bin/obj_localization: /usr/lib/libOpenNI2.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libexpat.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libjpeg.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpng.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libtiff.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libgl2ps.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_io.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libflann_cpp_s.a
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_search.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_features.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libqhull.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpcl_people.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libfreetype.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libpython2.7.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libproj.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libnetcdf.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libtheoraenc.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libtheoradec.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libogg.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libxml2.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/hdf5/openmpi/libhdf5.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libsz.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libz.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libdl.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libm.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
../bin/obj_localization: /usr/local/lib/libceres.a
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.1
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libtbb.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/liblapack.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libf77blas.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libatlas.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/librt.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/liblapack.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libf77blas.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libatlas.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/librt.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libGLU.so
../bin/obj_localization: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/obj_localization: /usr/local/lib/libOpenMeshCore.so
../bin/obj_localization: CMakeFiles/obj_localization.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/iaslab/TesiRC/d2co/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/obj_localization"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/obj_localization.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/obj_localization.dir/build: ../bin/obj_localization

.PHONY : CMakeFiles/obj_localization.dir/build

CMakeFiles/obj_localization.dir/requires: CMakeFiles/obj_localization.dir/apps/obj_localization.cpp.o.requires
CMakeFiles/obj_localization.dir/requires: CMakeFiles/obj_localization.dir/apps/apps_utils.cpp.o.requires

.PHONY : CMakeFiles/obj_localization.dir/requires

CMakeFiles/obj_localization.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/obj_localization.dir/cmake_clean.cmake
.PHONY : CMakeFiles/obj_localization.dir/clean

CMakeFiles/obj_localization.dir/depend:
	cd /home/iaslab/TesiRC/d2co/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/iaslab/TesiRC/d2co /home/iaslab/TesiRC/d2co /home/iaslab/TesiRC/d2co/build /home/iaslab/TesiRC/d2co/build /home/iaslab/TesiRC/d2co/build/CMakeFiles/obj_localization.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/obj_localization.dir/depend

