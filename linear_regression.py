#  Peter Stephens
#  5/23/2016

#  Performs the Chi-squared test on Lending Club Data and prints the result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
import collections
import subprocess

#  Clean the directory of old png files
proc = subprocess.check_call("rm -rf *.png",  shell=True)

#  Read in Lending Club Data form git hub repository
loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')

#  Clean Data:  Remove null value rows
loansData.dropna(inplace=True)

loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: float(x.rstrip('%')))
loansData['Loan.Length']   = loansData['Loan.Length'].map(lambda x: int(x.rstrip('months')))
loansData['FICO.Score']    = loansData['FICO.Range'].map(lambda x: int(x.split('-')[0]))

#  Create Histogram of FICO scores 
plt.figure()
a = loansData['FICO.Score'].hist()
plt.savefig("Bar_Plot_FICO_Score.png")

#  Create Scatter Matrix of loan data
plt.figure()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.savefig("Scatter_Matrix_Loan_Data.png")

#  Create Scatter Plot of loan data (FICO vs Interest Rate)
plt.figure()
a = loansData.plot.scatter(x = 'FICO.Score', y = 'Interest.Rate')
plt.savefig("Scatter_Plot_Loan_Data.png")

# The dependent variable
y = np.matrix(loansData['Interest.Rate']).transpose()

# The independent variables shaped as columns
x1 = np.matrix(loansData['FICO.Score']).transpose()
x2 = np.matrix(loansData['Amount.Requested']).transpose()
x = np.column_stack([x1,x2])

#  Create a linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

#  Output the Regression Summary
print(f.summary())

