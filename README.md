#Machine learning project 1: By Lamiae Omari Alaoui & Nessreddine Loudiy


Full project description can be found [here](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf).  

The project is organised into two folders:

- scripts: The folder that contains our codes. It has three files:
	- proj1_helpers.py: the same as provided in the [github repository](https://github.com/epfml/ML_course/blob/master/projects/project1/scripts/proj1_helpers.py).
	- implementations.py: Contain all the algorithms we used to solve the problem.
	- run.py: A script that generates the output, given the test dataset.
- data: The folder that contains the training and test data, and this is where we will generate our output. It should initially contain:
	- train.csv: Training data.
	- test.csv: Test data.

To generate the same output that we got:

1. Create two folders under the same directory, **scripts** and **data**.
2. Download the zipped data from this [link](https://github.com/epfml/ML_course/tree/master/projects/project1/data) and unzip them into **data** folder.
3. Download [proj1_helpers.py](https://github.com/epfml/ML_course/blob/master/projects/project1/scripts/proj1_helpers.py) and put it in scripts folder.
4. Put the **implementations.py** and **run.py** files that we uploaded in scripts folder.
5. Open terminal, navigate to scripts folder, please make sure python is added to your path, and execute:

		>>> python run.py
		
6. Wait until the code ends executing and you shall see a new file **output_Lamiae_Nessreddine.csv** created under data folder. You could now use it in the [AICrowd plateform](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019).
		