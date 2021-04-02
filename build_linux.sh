#!/bin/bash
container_name=$(basename $(dirname $(realpath $0)))
echo "Building seunglab/$container_name"
docker build . -f manylinux1.Dockerfile --tag "seunglab/$container_name:manylinux1"
docker build . -f manylinux2010.Dockerfile --tag "seunglab/$container_name:manylinux2010"
docker build . -f manylinux2014.Dockerfile --tag "seunglab/$container_name:manylinux2014"
docker run -v $PWD/dist:/output "seunglab/$container_name:manylinux1" /bin/bash -c "cp -r wheelhouse/* /output"
docker run -v $PWD/dist:/output "seunglab/$container_name:manylinux2010" /bin/bash -c "cp -r wheelhouse/* /output"
docker run -v $PWD/dist:/output "seunglab/$container_name:manylinux2014" /bin/bash -c "cp -r wheelhouse/* /output"