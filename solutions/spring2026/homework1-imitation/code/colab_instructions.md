## Running the homework on Colab

1. Upload the `hw1` folder to Google Drive.
2. Go to [https://colab.research.google.com](https://colab.research.google.com/) and create a new notebook. Select a **GPU runtime** (Runtime → Change runtime type → T4 GPU).
3. Mount your Google Drive and navigate to the project:
	```python
	from google.colab import drive
	drive.mount('/content/gdrive')
	%cd /content/gdrive/MyDrive/hw1
	```
	(Adjust the path if you placed the folder elsewhere.)
4. Install only the packages Colab doesn't already have (torch, numpy, matplotlib are pre-installed — reinstalling them can break things):
	```
	!pip install gymnasium pygame "imageio[ffmpeg]"
	```
	If you get `pip: command not found`, use `!python -m pip install gymnasium pygame "imageio[ffmpeg]"` instead.
5. Verify everything works:
	```
	!python -c "import torch; import gymnasium; import pygame; print('All good')"
	```
6. Locate the "Files" tab on the left sidebar to browse and edit the Python files (`networks.py`, `losses.py`, `dagger.py`, etc.).
7. Run individual parts as you implement them:
	```
	!python main.py --method bc_reg --env easy
	!python main.py --method bc_reg --env hard
	!python main.py --method bc_flow --env hard
	!python main.py --method dagger --env hard
	```
	Or run everything at once:
	```
	!python main.py
	```
8. Outputs are saved under timestamped subdirectories:
	- `plots/` — videos (`.mp4`) and comparison charts (`.png`)
	- `models/` — trained model checkpoints (`.pt`)
	- `results/` — text summaries with trajectory logs (`.txt`)
