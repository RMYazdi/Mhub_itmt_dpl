
# Deploying the ITMT Deep Learning Model on MHub
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8361032.svg)](https://doi.org/10.5281/zenodo.8361032)

---

## Quick Start

### Prerequisites
- **Docker**: [Get Docker](https://www.docker.com/get-started).
- **NVIDIA Toolkit** (for GPU acceleration): [NVIDIA Toolkit](https://developer.nvidia.com/cuda-downloads).
- **Git**: [Get Git](https://git-scm.com/downloads).
- **s5cmd**: Install `s5cmd` ([Installation Guide](https://github.com/peak/s5cmd#installation)).
- **Access to `tmt2` Repository**: Clone from [https://github.com/RMYazdi/tmt2](https://github.com/RMYazdi/tmt2).

---

## Step-by-Step Guide

### 1. Set Up Environment Variables

```bash
export TUTORIAL_DIR="$HOME/mhub_tutorial_tmt2"
export MODELS_FORK_REPO="https://github.com/<your_github_username>/models.git"  # Forked repo URL
export MODELS_FORK_DIR="$TUTORIAL_DIR/mhub_models"
export DATA_DIR="$TUTORIAL_DIR/data"
export MODEL_NAME="tmt2_model"
export MODEL_BASE_DIR="$MODELS_FORK_DIR/models/$MODEL_NAME"
export DOCKER_IMAGE="mhubai-dev/$MODEL_NAME"

mkdir -p $TUTORIAL_DIR
```

### 2. Fork and Clone MHubAI Models Repository

1. **Fork** [MHubAI/models](https://github.com/mhubai/models).
2. **Clone** the forked repo:

```bash
git clone $MODELS_FORK_REPO $MODELS_FORK_DIR
```

### 3. Create Model Directory Structure

```bash
mkdir -p $DATA_DIR/input $DATA_DIR/output
mkdir -p $MODEL_BASE_DIR/{config,dockerfiles,utils,src}
touch $MODEL_BASE_DIR/config/test.yml
touch $MODEL_BASE_DIR/dockerfiles/Dockerfile
touch $MODEL_BASE_DIR/utils/Thresholder.py
```

### 4. Create a New Branch in Forked Repository

```bash
cd $MODELS_FORK_DIR
git checkout -b $MODEL_NAME
```

### 5. Download Sample MRI Data

```bash
mkdir -p $DATA_DIR/input
s5cmd --no-sign-request --endpoint-url https://s3.amazonaws.com cp 's3://idc-open-data/1e05db78-9310-4ae3-b0ae-47c0cc9cf8a2/*' $DATA_DIR/input
```

### 6. Clone the `tmt2` Repository Locally

```bash
git clone https://github.com/RMYazdi/tmt2.git $TUTORIAL_DIR/tmt2
```

### 7. Place `tmt2` Inside the Model Directory

```bash
cp -r $TUTORIAL_DIR/tmt2 $MODEL_BASE_DIR/
```

### 8. Write and Adjust Dockerfile

```dockerfile
# $MODEL_BASE_DIR/dockerfiles/Dockerfile

# Use the MHub base image
FROM mhubai/base:latest

ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update &&     apt-get install --no-install-recommends -y     build-essential software-properties-common     tzdata ffmpeg libsm6 libxext6 &&     add-apt-repository -y ppa:deadsnakes/ppa &&     apt-get install --no-install-recommends -y     python3.9 python3.9-distutils python3-pip python3.9-dev &&     rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
RUN python3 -m pip install --upgrade pip

# Install Python packages
RUN pip3 install --no-cache-dir     tensorflow==2.10     nilearn pandas     scipy numpy matplotlib tqdm imageio scikit-image     scikit-learn itk-elastix SimpleITK nibabel 'intensity-normalization[ants]'     wandb jupyter opencv-python

# Copy the local tmt2 repository into the Docker container
COPY ./tmt2 /app/tmt2

ENV PYTHONPATH=/app/tmt2:$PYTHONPATH
ARG MHUB_MODELS_REPO
RUN buildutils/import_mhub_model.sh tmt2_model ${MHUB_MODELS_REPO}

ENTRYPOINT ["mhub.run"]
CMD ["--config", "/app/models/tmt2_model/config/test.yml"]
```

### 9. Create MHub Model Runner (`Thresholder.py`)

```python
# $MODEL_BASE_DIR/utils/Thresholder.py

from mhubio.core import Module, Instance, InstanceData, IO
import os

@IO.Config('age', float, 0.0, the='age of the subject')
@IO.Config('gender', str, 'M', the='gender of the subject (M/F)')
class Thresholder(Module):
    age: float
    gender: str

    @IO.Instance()
    @IO.Input('image', 'nifti:mod=mr', the='input MRI scan')
    @IO.Output('thresholded', 'thresholded.nii.gz', 'nifti:mod=seg:model=tmt2:roi=SEGMENTATION', the='segmented image')
    def task(self, instance: Instance, image: InstanceData, thresholded: InstanceData) -> None:
        cmd = [
            'python3', '/app/tmt2/main.py',
            '--input_path', image.abspath,
            '--output_path', thresholded.abspath,
            '--age', str(self.age),
            '--gender', self.gender
        ]
        self.log.debug('Running command: %s', ' '.join(cmd))
        self.subprocess(cmd)
```

### 10. Create Workflow Configuration (`test.yml`)

```yaml
# $MODEL_BASE_DIR/config/test.yml

general:
  data_base_dir: /app/data
  version: 1.0
  description: "Configuration for tmt2 MRI segmentation model"

execute:
- DicomImporter
- NiftiConverter
- Thresholder
- DataOrganizer

modules:
  DicomImporter:
    meta:
      mod: '%Modality'

  NiftiConverter:
    in_datas: dicom:mod=mr
    engine: dcm2niix

  Thresholder:
    age: 25.0  # Adjust as needed
    gender: 'F'  # Adjust as needed

  DataOrganizer:
    targets:
      - nifti:mod=seg-->[i:sid]/segmentation.nii.gz
```

### 11. Build Docker Image

```bash
docker build -t $DOCKER_IMAGE     --build-arg MHUB_MODELS_REPO=$MODELS_FORK_REPO::$MODEL_NAME     -f $MODEL_BASE_DIR/dockerfiles/Dockerfile     $MODEL_BASE_DIR
```

### 12. Test Docker Image and MHub Workflow

```bash
docker run --rm -it --gpus all   -v $MODEL_BASE_DIR:/app/models/$MODEL_NAME   -v $DATA_DIR/input:/app/data/input_data:ro   -v $DATA_DIR/output:/app/data/output_data   $DOCKER_IMAGE --cleanup --debug --model $MODEL_NAME --config /app/models/$MODEL_NAME/config/test.yml
```

### 13. Modify Configuration Without Rebuilding

Modify `test.yml` as needed:

```bash
nano $MODEL_BASE_DIR/config/test.yml
```

Rerun:

```bash
docker run --rm -it --gpus all   -v $MODEL_BASE_DIR:/app/models/$MODEL_NAME   -v $DATA_DIR/input:/app/data/input_data:ro   -v $DATA_DIR/output:/app/data/output_data   $DOCKER_IMAGE --cleanup --debug --model $MODEL_NAME --config /app/models/$MODEL_NAME/config/test.yml
```

### 14. Commit and Push Changes

```bash
cd $MODELS_FORK_DIR
git add models/$MODEL_NAME/
git commit -m "Add tmt2 model implementation with updated workflows"
git push --set-upstream origin $MODEL_NAME
```

### 15. Submit Your Model to MHub (Optional)

Create a pull request from your fork to the `MHubAI/models` repository.



**Note**: Replace `<your_github_username>` in the environment variables and repository URLs with your GitHub username.
