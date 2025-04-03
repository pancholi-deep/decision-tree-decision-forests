
Python Programming Language (Version 3.9.13)

Code Structure:
1. Importing Libraries
2. Load data
3. train_decision_tree function 
	to train a decision tree based on the specified option
4. test_decision_tree function 
	to test a trained decision tree on the test data and print the results,
	calculates accuracy for each test object and prints the details
5. Main section
	read command-line arguments for the training file, test file, and the option,
	load training and test data using the load_data function,
	trains a decision tree using the train_decision_tree function,
	tests the decision tree on the test data using the test_decision_tree function.

How to run the code:
command: python dtree.py training_file.txt test_file.txt optimized
example: python dtree.py pendigits_training.txt pendigits_test.txt optimized