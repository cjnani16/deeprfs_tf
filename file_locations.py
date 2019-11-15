import socket
PATH_TO_IMDB_GENDER = '[unknown]'
PATH_TO_CHECKPOINTS = '[unknown]'

if (socket.gethostname() == 'AMOXICILLIN'):
	PATH_TO_IMDB_GENDER = 'Z:/IMDB_Gender/'
	PATH_TO_CHECKPOINTS = 'Z:/checkpoints_vgg16/'

if (socket.gethostname() == 'nyla'):
	PATH_TO_IMDB_GENDER = '/biac2/kgs/projects/deepRFs/IMDB_Gender/'
	PATH_TO_CHECKPOINTS = '/biac2/kgs/projects/deepRFs/checkpoints_vgg16/'