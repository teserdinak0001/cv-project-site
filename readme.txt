README

0. Extract the "input" compressed zip folder to the main project directory
	a. main project directory is the top level that contains pipeline.py

1. Requirements to run this pipeline:
	a. Python 3.11.9
	b. Dependencies in requirements.txt
	c. Folder named "input" with .mp4 files corresponding to this project
	
2. Steps to run the pipeline:
	a. Download files from GitHub: https://github.com/teserdinak0001/cv-project-site
	b. Install python
	c. Create a project folder and initialize a new virtual environment
		- cd "<project folder>"
		- python -m venv .venv
		- .venv\Scripts\Activate.ps1
	d. Install dependencies from requirements.txt
		- pip install -r requirements.txt
	e. Open pipeline.py and run via the following command
		- python pipeline.py
		
3. Code execution can take a substantial amount of time, most of which is data extraction. Alternatively,
the data extraction task can be skipped and the already extracted data may be used by running segmentation.py
then learning.py.
	a. python segmentation.py
	b. python learning.py

4. After the code hasn't finished executing final results can be found in:
	a. cluster_eval_merged.csv
	b. clusters_merged/
		- cluster_<id>
