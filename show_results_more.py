from importer import *
from tqdm import tqdm
import cv2
parser = argparse.ArgumentParser()

# Machine Details
parser.add_argument('--gpu', type=str, required=True, help='[Required] GPU to use')


# parser.add_argument('--data_dir', type=str, required=True, help='Path to shapenet rendered images')

# Experiment Details
parser.add_argument('--mode', type=str, required=True, help='[Required] Latent Matching setup. Choose from [lm, plm]')
parser.add_argument('--exp', type=str, required=True, help='[Required] Path of experiment for loading pre-trained model')
parser.add_argument('--load_best', action='store_true', help='load best val model')

# AE Details
parser.add_argument('--bottleneck', type=int, required=False, default=512, help='latent space size')
# parser.add_argument('--bn_encoder', action='store_true', help='Supply this parameter if you want bn_encoder, otherwise ignore')
parser.add_argument('--bn_decoder', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')
parser.add_argument('--bn_decoder_final', action='store_true', help='Supply this parameter if you want bn_decoder, otherwise ignore')

parser.add_argument('--eval_set', type=str, help='Choose from train/valid')
parser.add_argument('--image', type=str, help='Choose from train/valid')
parser.add_argument('--txt', type=str, help='Choose from train/valid')

# Other Args
parser.add_argument('--visualize', action='store_true', help='supply this parameter to visualize')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

if FLAGS.visualize:
	BATCH_SIZE = 1

NUM_POINTS = 2048
NUM_VIEWS = 24
HEIGHT = 128
WIDTH = 128
PAD = 35
INPUT = FLAGS.image

if FLAGS.visualize:
	from utils.show_3d import show3d_balls
	ballradius = 3






if __name__ == '__main__':

	ip_image = cv2.imread(INPUT)[4:-5, 4:-5, :3]
	ip_image = cv2.resize(ip_image,(128,128),interpolation=cv2.INTER_CUBIC)
	ip_image = cv2.cvtColor(ip_image, cv2.COLOR_BGR2RGB)
	batch_ip = []
	batch_ip.append(ip_image)
	batch_ip = np.array(batch_ip)
	img_inp = tf.placeholder(tf.float32, shape=(1,HEIGHT, WIDTH, 3), name='img_inp')
	
	# Generate Prediction
	with tf.variable_scope('psgn_vars'):
		z_latent_img = image_encoder(img_inp, FLAGS)
			
	with tf.variable_scope('pointnet_ae') as scope:
		out_img = decoder_with_fc_only(z_latent_img, layer_sizes=[256,256,np.prod([NUM_POINTS, 3])],
			b_norm=FLAGS.bn_decoder,
			b_norm_finish=False,
			verbose=True,
			scope=scope
			)
		reconstr_img = tf.reshape(out_img, (BATCH_SIZE, NUM_POINTS, 3))

	# Perform Scaling
	reconstr_img_scaled = scale_image(reconstr_img)
	#reconstr_img_scaled = reconstr_img
	
	if FLAGS.load_best:
		snapshot_folder = join(FLAGS.exp, 'best')
	else:
		snapshot_folder = join(FLAGS.exp, 'snapshots')

	 # GPU configuration
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		load_previous_checkpoint(snapshot_folder, saver, sess, is_training=False)
		tflearn.is_training(False, session=sess)

		if FLAGS.eval_set == 'valid':
			_pred_scaled = sess.run(reconstr_img_scaled,feed_dict={img_inp:batch_ip})
			if FLAGS.visualize:
				for i in xrange(BATCH_SIZE):
					cv2.imshow('', batch_ip[0])
					print 'Displaying Pr scaled icp 1k'
					print _pred_scaled
					show3d_balls.showpoints(_pred_scaled[i], ballradius=3)
			

		else:
			print 'Invalid dataset. Choose from [shapenet, pix3d]'
			sys.exit(1)
