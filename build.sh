
if [ -d "lib/" ]; then rm -rf "lib/"; fi
if [ -d "out/" ]; then rm -rf "out/"; fi
if [ -d "build/" ]; then rm -rf "build/"; fi

mkdir lib/
mkdir out/
mkdir build/

cd build/

cmake ..
make -j
