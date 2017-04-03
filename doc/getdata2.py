import sys

fname = sys.argv[1]

with open(fname) as f:
    content = f.readlines()

content = [x.strip() for x in content] 

epoch = ""
trainAcc = ""
tuneAcc = ""
testAcc = ""

print "newcurve marktype none linetype dotted label : Train set accuracy"
print "pts"
for line in content:
	if line.startswith('Train size:'):		
		eopch = line.split(":")[1]
		
	if line.startswith('Train accuracy:'):
		trainAcc = line.split(':')[1]		
		print eopch, trainAcc
		
		

print "newcurve marktype none linetype dashed color 0 1 0 label : Tune set accuracy"
print "pts"
for line in content:
	if line.startswith('Train size:'):		
		eopch = line.split(":")[1]

	if line.startswith('Tune accuracy:'):
		tuneAcc = line.split(':')[1]
		print eopch, tuneAcc

print "newcurve marktype none linetype solid color 0 0 1 label : Test set accuracy"
print "pts"
for line in content:
	if line.startswith('Train size:'):		
		eopch = line.split(":")[1]

	if line.startswith('Test accuracy:'):
		testAcc = line.split(':')[1]		
		print eopch, testAcc

		
