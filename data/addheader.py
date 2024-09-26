import os

# add a header for yoochoose dataset
with open('yoochoose-clicks.dat', 'r') as f, open('yoochoose-clicks-withHeader.dat', 'w') as fn:
    fn.write('sessionId,timestamp,itemId,category'+'\n')
    for line in f:
        fn.write(line)