import argparse
import os
import tensorflow as tf

tf.set_random_seed(20)
from model import vae

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='train_100', help='path of the dataset') # 200411_HKC_3rd
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=1000, help='# of epoch to decay lr')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam') # default=0.0002
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='reconstruction', help='train, test, reconstruction') # test : generator, test_d : discriminator
parser.add_argument('--save_freq', dest='save_freq', type=int, default=500, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--alpha', dest='alpha', type=float, default=1000.0, help='weight on regression loss')
parser.add_argument('--latent_dims', dest='latent_dims', type=int, default=5, help='latent space dimension')
parser.add_argument('--beta', dest='beta', type=int, default=100, help='weight on kl div loss')



args = parser.parse_args()


def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = vae(sess, args)
        if args.phase == 'train':
            model.train(args)
        elif args.phase == 'test':
            model.test(args)
        elif args.phase == 'reconstruction':
            model.test_reconstruction(args)

if __name__ == '__main__':
    tf.app.run()
    #try:
    #    tf.app.run()
    #    print('end')
    #except:
    #    pass
    