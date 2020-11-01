FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
LABEL maintainer="Lloyd Hughes <hughes.lloyd@gmail.com>"

##############################################################################
# Upgrade conda, pip and apt-get
##############################################################################
#RUN conda update conda -y --quiet
#RUN conda install -y pip
#RUN pip install --upgrade pip
RUN apt-get update

RUN apt-get install -y libfftw3-dev libsm6 libxext6 libxrender-dev

# RUN apt-get install -y libgl1-mesa-glx

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

#COPY environment.yml /root/environment.yml
#RUN conda update -y -n base -c defaults conda
# RUN conda create -n custom python=3.7 numpy scipy scikit-learn 
# RUN conda env update -f /root/environment.yml

#RUN echo "source activate custom" > ~/.bashrc
#ENV PATH /opt/conda/envs/env/bin:$PATH

# RUN /bin/bash -c "source activate custom && conda install -y pytorch-nightly -c pytorch"
# RUN /bin/bash -c "source activate custom && conda install -y pytorch=1.1 cuda90 torchvision -c pytorch"
RUN conda install -y -c menpo opencv3
RUN conda install -y dask=0.19.3 scikit-learn
RUN conda install -y -c conda-forge geopandas=0.5.0 libspatialite=4.3.0a libspatialindex=1.9.0
RUN pip install pytorch-ignite==0.2.0 rasterio==1.0.8 tensorboardX torchsummary dotmap==1.3.4 pandas==0.23.4 pyproj==2.1.0 geojson utm==0.4.2 fiona==1.8.0 
RUN pip install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio
RUN pip install imgaug visdom

WORKDIR /src
#RUN pip uninstall -y apex || :
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /src/apex
RUN python setup.py install
#RUN /bin/bash -c "source activate custom && python setup.py install" 


COPY . /src
WORKDIR /src

ENTRYPOINT ["python"]

# FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
# LABEL maintainer="Lloyd Hughes <hughes.lloyd@gmail.com>"

# ##############################################################################
# # Upgrade conda, pip and apt-get
# ##############################################################################
# RUN conda update conda -y --quiet
# RUN pip install --upgrade pip
# RUN apt-get update

# RUN apt-get install -y libfftw3-dev

# ENV LC_ALL=C.UTF-8
# ENV LANG=C.UTF-8

# COPY environment.yml /root/environment.yml
# # RUN conda update -y -n base -c defaults conda
# RUN conda env create -f /root/environment.yml -n custom
# # RUN conda env update -f /root/environment.yml

# RUN echo "source activate custom" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH

# # RUN /bin/bash -c "source activate custom && conda install -y pytorch-nightly -c pytorch"
# RUN /bin/bash -c "source activate custom && conda install -y pytorch=1.1 cuda90 torchvision -c pytorch"

# WORKDIR /src
# RUN pip uninstall -y apex || :
# RUN git clone https://github.com/NVIDIA/apex.git
# WORKDIR /src/apex
# RUN /bin/bash -c "source activate custom && python setup.py install" 

# COPY . /src
# WORKDIR /src

# ENTRYPOINT ["/opt/conda/envs/custom/bin/python"]
