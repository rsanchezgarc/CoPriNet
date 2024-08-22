FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install essential packages
RUN apt-get update && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone the CoPriNet repository
RUN git clone https://github.com/oxpig/CoPriNet.git

# Change to the CoPriNet directory
WORKDIR /CoPriNet

# Create the conda environment using the provided .yml file
RUN conda env create -f CoPriNet_env.yml
RUN conda init

SHELL ["conda", "run", "--no-capture-output", "-n", "CoPriNet", "/bin/bash", "-c"]

RUN mkdir /inputDir
RUN mkdir /oututDir
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "CoPriNet", "python", "-m", "pricePrediction.predict.predict"]

# docker build -t coprinet:latest . # --no-cache

#TO RUN COPRINET
# docker run --gpus all -v /host/inputDir:/inputDir -v /host/outputDir:/outputDir -it coprinet:latest -o /outputDir/preds.csv /inputDir/input.csv

# docker run --gpus all -it --entrypoint /bin/bash coprinet:latest
# conda activate CoPriNet
# python -m pricePrediction.predict.predict data/testData/npnp_dataset.csv

