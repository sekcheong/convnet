import sys

fname = sys.argv[1]

with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 

epoch = ""
trainAcc = ""
tuneAcc = ""
testAcc = ""

for line in content:
	if line.startswith('Epoch:'):		
		eopch = line.split(":")[1]
		
	if line.startswith('Train accuracy:'):
		trainAcc = line.split(':')[1]
		
	if line.startswith('Tune accuracy:'):
		tuneAcc = line.split(':')[1]
		
	if line.startswith('Test accuracy:'):
		testAcc = line.split(':')[1]		
		print eopch, testAcc

