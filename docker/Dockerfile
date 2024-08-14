FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3
ARG CUDA_ARCH
RUN /bin/bash -c 'apt-get update; \
        apt-get install -y vim tmux python-is-python3 htop ninja-build tree; \
        pip install -v pccm wheel SharedArray scikit-learn scikit-image av2 kornia easydict tensorboardX'

RUN /bin/bash -c 'cd; \
        git clone -b v0.4.11 https://github.com/FindDefinition/cumm.git; \
        cd cumm; \
        MAX_JOBS=$(nproc) CUMM_CUDA_ARCH_LIST=$CUDA_ARCH CUMM_DISABLE_JIT="1" python setup.py bdist_wheel; \
        pip install dist/*'

RUN /bin/bash -c 'cd; \
        git clone -b v2.3.6 https://github.com/traveller59/spconv.git; \
        cd spconv; \
        MAX_JOBS=$(nproc) CUMM_CUDA_ARCH_LIST=$CUDA_ARCH SPCONV_DISABLE_JIT="1" python setup.py bdist_wheel; \
        pip install dist/*'

RUN /bin/bash -c 'cd; \
        git clone -b 2.1.2 https://github.com/rusty1s/pytorch_scatter.git; \
        cd pytorch_scatter; \
        CPATH=$CPATH:/usr/local/cuda/include TORCH_CUDA_ARCH_LIST=$CUDA_ARCH FORCE_CUDA=1 pip install -v .'

RUN /bin/bash -c 'cd; \
        git clone https://github.com/CSL-KU/VALO.git; \
        cd VALO; \
        MAX_JOBS=$(nproc) TORCH_CUDA_ARCH_LIST=$CUDA_ARCH python setup.py develop'

RUN /bin/bash -c 'cd; \
	git clone -b v1.1.2 https://github.com/nutonomy/nuscenes-devkit.git; \
	cd nuscenes-devkit; \
	sed -i -z "s/opencv-python\n//g" setup/requirements/requirements_base.txt; \
	pip install -r setup/requirements.txt; \
	pip install motmetrics; \
	echo "export PYTHONPATH=\${PYTHONPATH}:\$HOME/nuscenes-devkit/python-sdk" >> ~/.bashrc; \
	echo "export LD_PRELOAD=\${LD_PRELOAD}:/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0" >> ~/.bashrc; \
	source ~/.bashrc'

RUN /bin/bash -c 'apt purge opencv*; \
	pip install numpy==1.23.0; \
	pip install kornia==0.6.12; \
	pip install --upgrade matplotlib'
