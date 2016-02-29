import csv
import numpy as np
import scipy as sp

import prettyplotlib as ppl
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split

from collections import defaultdict

def process_data(base='./data/o', maxval=14):
	letters = ['A','m','1','1','5','R','o','%']
	lines = [0,3,6,9,12,15,18,21]
	cl = {l:i for i,l in enumerate(lines)}
	output = []
	for i in xrange(1,maxval+1):
		fname = base + ('0'*(3-len(str(i)))) + str(i) + ".csv"
		with open(fname, 'r') as f:
			user = defaultdict(lambda:[])
			for k,line in enumerate(f):
				if k in cl:
					letter = letters[cl[k]]
					vals = line.strip().split(",")
					even = True
					for j,v in enumerate(vals):
						if even:
							even = False
						else:
							if len(vals[j-1]) > 0 and vals[j-1].split(".")[1] == letter:
								user[j].append(v)
							else:
								user[j].append(None)
							even = True

			for k,v in user.iteritems():
				if None not in v:
					output.append((i,np.array([int(vval) for vval in v])))

	return zip(*output)[::-1]

class GMM:
	def __init__(self, reduction=1, do_pca=False):
		self.means = []
		self.vars = []
		self.user_ids = {}
		self.pca = PCA(2)
		self.do_pca = do_pca
		self.reduction = reduction

	def fit(self, xdata, ydata):
		if self.do_pca:
			xx, xy = zip(*self.pca.fit_transform(xdata))
			xdata = np.array(zip(xx,np.array(xy)/self.reduction))
		users = np.unique(ydata)

		for u in users:
			curdata = xdata[ydata == u]
			self.user_ids[len(self.means)] = u
			self.means.append(np.mean(curdata, 0))
			self.vars.append(np.sqrt(np.var(curdata, 0)))

	def __compute_closest(self, xval):
			return self.user_ids[np.argmax([np.sum(np.abs((xval-self.means[i])/self.vars[i])) for i in xrange(len(self.means))])]

	def predict(self, xdata):
		if self.do_pca:
			xx, xy = zip(*self.pca.fit_transform(xdata))
			xdata = np.array(zip(xx,np.array(xy)/self.reduction))
		return np.array([self.__compute_closest(x) for x in xdata])


class SKLData:
	def __init__(self, fname, funcs, header=True, test_size=0.1):
		self.data = []
		self.clf = None

		self.__noutput = len(funcs)

		with open(fname, 'r') as f:
			csf = csv.reader(f)

			if header:
				csf.next()

			for row in csf:
				try:
					self.data.append([fc(row) for fc in funcs])
				except ValueError:
					pass

		if self.__noutput >= 2:
			self.X, self.y = zip(*self.data)[:2]
			self.feats = zip(*self.data)[2:]
			
			self.X = np.array(self.X)
			self.y = np.array(self.y)

			self.regenerate(test_size)

	def __score(self, clf=None, xtest=None, ytest=None):
		if xtest is None:
			xtest = self.Xtest
			ytest = self.ytest

		if clf is None:
			if self.clf is not None:
				return np.sum(self.clf.predict(xtest) == ytest)/float(len(ytest))
			else:
				raise ValueError("No Classifier Passed In")
		else:
			return np.sum(clf.predict(xtest) == ytest)/float(len(ytest))

	def transform(self, lx, ly):
		self.X = lx(self.X)
		self.y = ly(self.y)
		self.regenerate()

	def regenerate(self, test_size=0.1, X=None, y=None):
		if X is None:
			X = self.X
			y = self.y

		self.Xtrain, self.Xtest, self.ytrain, self.ytest = skl.cross_validation.train_test_split( \
				X, y, test_size=test_size)

		try:
			self.Xpca = skl.decomposition.PCA(2).fit_transform(X)
		except ValueError:
			self.Xpca = None
			print "PCA Failed - Probably Different X lengths?"
			print "Distinct X-Length Values:"
			print set([len(self.data[i][0]) for i in xrange(len(self.data))])


	def train(self, clf, lfunc=None, test_size=0.1):
		self.clf = clf
		if lfunc is not None:
			X = lfunc(self.X)
			xtrain, xtest, ytrain, ytest = skl.cross_validation.train_test_split(X, self.y, test_size=test_size)
			self.clf.fit(xtrain, ytrain)
			return self.__score(xtest=xtest, ytest=ytest)
		else:
			self.clf.fit(self.Xtrain, self.ytrain)
			return self.__score()

	def __scatter(self, data, **kwargs):
		xnew, ynew = zip(*data)
		return plt.scatter(xnew, ynew, **kwargs)

	def plot(self, subject, **kwargs):
		if self.Xpca is None:
			raise ValueError("PCA Failed Earlier!")
		return self.__scatter(self.Xpca[self.y == subject], label='Subject '+str(subject), **kwargs)


def plot(x, y, **kwargs):
	ppl.scatter(x, y, True, **kwargs)