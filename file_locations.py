import socket
PATH_TO_IMDB_AGE = '[unknown]'
PATH_TO_CHECKPOINTS = '[unknown]'
PATH_TO_VPNL_ID = '[unknown]'
PATH_TO_VPNL_GENDER = '[unknown]'

if (socket.gethostname() == 'AMOXICILLIN'):
	PATH_TO_IMDB_AGE = 'Z:/imdb_age/'
	PATH_TO_CHECKPOINTS = 'Z:/checkpoints_vgg16/'
	PATH_TO_VPNL_ID = 'Z:/FaceDataCp'
	PATH_TO_VPNL_GENDER = 'Z:/FaceDataGender'

if (socket.gethostname() == 'nyla' or socket.gethostname() == 'strelka'):
	PATH_TO_IMDB_AGE = '/biac2/kgs/projects/deepRFs/imdb_age/'
	PATH_TO_CHECKPOINTS = '/biac2/kgs/projects/deepRFs/checkpoints_vgg16/'
	PATH_TO_VPNL_ID = '/biac2/kgs/projects/deepRFs/FaceDataCp'
	PATH_TO_VPNL_GENDER = '/biac2/kgs/projects/deepRFs/FaceDataGender'
