## Install dependencies

### Local setup (conda)

1. Install conda if you don't already have it: [https://docs.conda.io/projects/conda/en/latest/user-guide/install/](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

2. Create and activate the environment:
	```bash
	conda create -n cs224r python=3.10 -y
	conda activate cs224r
	```

3. Install all dependencies with pip:
	```bash
	pip install torch gymnasium pygame matplotlib "imageio[ffmpeg]" "numpy==2.2.4"
	```

	For GPU support, follow the pip install command from [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

4. Verify:
	```bash
	python -c "import torch; import gymnasium; import pygame; print('All good')"
	```

### Google Colab

See [colab_instructions.md](colab_instructions.md).

## Troubleshooting

**`libgomp.so.1` not found / corrupted conda cache**
If `conda create` fails with missing library errors, clear the package cache and retry:
```bash
rm -rf "$(conda info --base)/pkgs/"*
conda create -n cs224r python=3.10 -y
```

**Pygame / SDL display errors**
```bash
export SDL_VIDEODRIVER=dummy
```

**ALSA/PulseAudio errors**
```bash
export SDL_AUDIODRIVER=dummy
```

**`imageio` can't find ffmpeg**
```bash
pip install "imageio[ffmpeg]"
```
