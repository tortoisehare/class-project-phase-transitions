import sys
import matplotlib.pyplot as plt

try:
    costfilename = sys.argv[1]
except:
    print '<cost file>'

with open(costfilename, 'r') as costfile:
    listline = costfile.readlines()

xlist = list()
ylist = list()
for i in listline:
    x,y = i.split(',')
    xlist.append(float(x))
    ylist.append(float(y))

plt.figure()
plt.plot(xlist, ylist)
plt.show()

