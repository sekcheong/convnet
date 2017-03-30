import sys

fname = sys.argv[1]

with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 

for line in content:
	epoch = ""
	trainAcc = ""
	tuneAcc = ""
	testAcc = ""
	if line.startswith('Epoch:'):		
		eopch = line.split(":")[1]
		print epoch, line.split(':')[1].strip()
	if line.startswith('Train accuracy:'):
		trainAcc = line.split(':')[1]
		print trainAcc
	if line.startswith('Tune accuracy:'):
		tuneAcc = line.split(':')[1]
		print tuneAcc
	if line.startswith('Test accuracy:'):
		testAcc = line.split(':')[1]		
		print testAcc

