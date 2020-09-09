#!/usr/local/bin/bash
docker build . -f manylinux1.Dockerfile --tag seunglab/cc3d:manylinux1
docker build . -f manylinux2010.Dockerfile --tag seunglab/cc3d:manylinux2010
docker build . -f manylinux2014.Dockerfile --tag seunglab/cc3d:manylinux2014
docker run -v $PWD/dist:/output seunglab/cc3d:manylinux1 /bin/bash -c "cp -r wheelhouse/* /output"
docker run -v $PWD/dist:/output seunglab/cc3d:manylinux2010 /bin/bash -c "cp -r wheelhouse/* /output"
docker run -v $PWD/dist:/output seunglab/cc3d:manylinux2014 /bin/bash -c "cp -r wheelhouse/* /output"