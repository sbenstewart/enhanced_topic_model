from os import path as path

class Stopwords():
	"""Stop words class given by mam"""
	def __init__(self, path):
		self.words = []
		with open(path, 'r') as f:
			for line in f:
				self.words.extend(line.split())

	def printword(self):
		"""Developed for testing purpose"""
		for i in self.words:
			print(i)

	def printall(self):
		"""Developed for testing purpose"""
		print(self.words)


def getStopWords():
	temp = path.join(path.split(path.abspath(sp.__file__))[0], 'long.txt')
	obj = Stopwords(temp)
	return obj.words

if __name__ == '__main__':

	obj = Stopwords()
	obj.printall()
	print(__file__)
	with open(os.path.abspath('long.txt'), 'r') as f:
		print(f.readline())
