# src/step0_encoding.py

import csv
import os
import sys

os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

report = open("reports/step0_encoding_report.txt", "w")
sys.stdout = report

data = open('data/Mutated_Orignal_biosequence.csv')
datacsvfile = csv.reader(data, delimiter=',')
sequenceData = list(datacsvfile)

# ===============================
# FIND MAX LENGTH
# ===============================
max_len = 0
for row in range(len(sequenceData)):
    seq_len = len(sequenceData[row][1])
    if seq_len > max_len:
        max_len = seq_len

print("Max Sequence Length:", max_len)

# ===============================
# WRITE OUTPUT
# ===============================
output = open('data/EIIPSTDADsequencesencoding.csv', 'w')
csvWriter = csv.writer(output, delimiter=',', lineterminator='\n')

for row in range(len(sequenceData)):
    sequencecode = []

    for item in range(len(sequenceData[row][1])):

        if sequenceData[row][1][item]=='L':
            sequencecode.append(0.0000)
        elif sequenceData[row][1][item]=='I':
            sequencecode.append(0.0000)
        elif sequenceData[row][1][item]=='N':
            sequencecode.append(0.0036)
        elif sequenceData[row][1][item]=='G':
            sequencecode.append(0.0050)
        elif sequenceData[row][1][item]=='V':
            sequencecode.append(0.0057)
        elif sequenceData[row][1][item]=='E':
            sequencecode.append(0.0058)
        elif sequenceData[row][1][item]=='P':
            sequencecode.append(0.0198)
        elif sequenceData[row][1][item]=='H':
            sequencecode.append(0.0242)
        elif sequenceData[row][1][item]=='K':
            sequencecode.append(0.0371)
        elif sequenceData[row][1][item]=='A':
            sequencecode.append(0.0373)
        elif sequenceData[row][1][item]=='Y':
            sequencecode.append(0.0516)
        elif sequenceData[row][1][item]=='W':
            sequencecode.append(0.0548)
        elif sequenceData[row][1][item]=='Q':
            sequencecode.append(0.0761)
        elif sequenceData[row][1][item]=='M':
            sequencecode.append(0.0823)	
        elif sequenceData[row][1][item]=='S':
            sequencecode.append(0.0829)
        elif sequenceData[row][1][item]=='C':
            sequencecode.append(0.0829)
        elif sequenceData[row][1][item]=='T':
            sequencecode.append(0.0941)
        elif sequenceData[row][1][item]=='F':
            sequencecode.append(0.0954)
        elif sequenceData[row][1][item]=='R':
            sequencecode.append(0.0956)
        elif sequenceData[row][1][item]=='D':
            sequencecode.append(0.1263)

    # ✅ PAD HERE (IMPORTANT FIX)
    if len(sequencecode) < max_len:
        sequencecode += [0] * (max_len - len(sequencecode))

    csvWriter.writerow([sequenceData[row][2], sequenceData[row][0]] + sequencecode)

output.close()
data.close()

print("Encoding Done Successfully")
report.close()