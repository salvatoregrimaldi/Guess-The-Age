================================================================
===================== FOLDER'S CONTENT =========================
================================================================

• training_100_100_100.py: 
	Training procedure with a classifier network made of three dense layers of 100 neurons each.
• training_100.py: 
	Training procedure with a classifier network made of only one dense layer of 100 neurons.
• k_fold.py: 
	K-fold cross validation.

• test.py: 
	Python script which takes as inputs a CSV file with the same format of the training annotations (--data) and the folder of the test images (--images) 
	and produces as output a CSV file with the predicted age for each image (--results). Thus, the script may be executed with the following command: 
	python test.py --data foo_test.csv --images foo_test/ --results foo_results.csv.
• test.ipynb: 
	A Google Colab Notebook which includes the commands for installing all the software requirements and executes the script test.py.
• requirements: 
	List of pip packages required to launch the python script "test.py"

• A3MoreDenseFold0_class: 
	The folder containing the submitted model.

• Report_Team7.pdf: 
	The report describing the proposed solution.