FROM quay.io/pypa/manylinux1_x86_64
MAINTAINER William Silversmith

ADD . /cc3d

WORKDIR "/cc3d"

ENV CC "g++"

RUN rm -rf *.so build __pycache__ dist 

RUN /opt/python/cp27-cp27m/bin/pip2.7 install pip --upgrade
RUN /opt/python/cp27-cp27m/bin/pip2.7 install numpy pytest 
RUN /opt/python/cp27-cp27m/bin/python2.7 setup.py develop
RUN /opt/python/cp27-cp27m/bin/python2.7 -m pytest -v -x automated_test.py

RUN /opt/python/cp34-cp34m/bin/pip3.4 install pip --upgrade
RUN /opt/python/cp34-cp34m/bin/pip3.4 install numpy==1.15.4 pytest
RUN /opt/python/cp34-cp34m/bin/python3.4 setup.py develop
RUN /opt/python/cp34-cp34m/bin/python3.4 -m pytest -v -x automated_test.py

RUN /opt/python/cp35-cp35m/bin/pip3.5 install pip --upgrade
RUN /opt/python/cp35-cp35m/bin/pip3.5 install numpy pytest
RUN /opt/python/cp35-cp35m/bin/python3.5 setup.py develop
RUN /opt/python/cp35-cp35m/bin/python3.5 -m pytest -v -x automated_test.py

RUN /opt/python/cp36-cp36m/bin/pip3.6 install pip --upgrade
RUN /opt/python/cp36-cp36m/bin/pip3.6 install numpy pytest
RUN /opt/python/cp36-cp36m/bin/python3.6 setup.py develop
RUN /opt/python/cp36-cp36m/bin/python3.6 -m pytest -v -x automated_test.py

RUN /opt/python/cp37-cp37m/bin/pip3.7 install pip --upgrade
RUN /opt/python/cp37-cp37m/bin/pip3.7 install numpy pytest
RUN /opt/python/cp37-cp37m/bin/python3.7 setup.py develop
RUN /opt/python/cp37-cp37m/bin/python3.7 -m pytest -v -x automated_test.py

RUN /opt/python/cp27-cp27m/bin/python2.7 setup.py sdist bdist_wheel
RUN /opt/python/cp34-cp34m/bin/python3.4 setup.py sdist bdist_wheel
RUN /opt/python/cp35-cp35m/bin/python3.5 setup.py sdist bdist_wheel
RUN /opt/python/cp36-cp36m/bin/python3.6 setup.py sdist bdist_wheel
RUN /opt/python/cp37-cp37m/bin/python3.7 setup.py sdist bdist_wheel

RUN for whl in `ls dist/*.whl`; do auditwheel repair $whl; done