#Assignment 1 - CS4552

Complete assignment can be built using cmake. 

**Assignment Configuration**

1. Change Between Double and Float version ***

    To change the build versions between float and double compile time parameters have to be changed.
    To do this go to main (top level) `CMakeLists.txt` file and change the line `16` option to `ON` or `OFF`
    
    eg: `option(USE_DOUBLE "Use double values in computation" ON)`
    
    This will build the double versions. 


**How to build the assignment**

Create a directory inside the assignment `(eg: build)`.
Go inside the directory and invoke the below commands sequentially.
 
 `cmake .. && make -j6`
 
 Now inside the build folder three folders will be created (Q1, Q2 and Q3), having
 the required executables inside each folder.
 
 ****NOTE : If you change the configuration in `A1Config.h.in` file you have to clean the build directiry and rebuild 
 from the beginning. This is due to the caching of the cmake*
  
