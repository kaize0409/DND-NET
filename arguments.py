import argparse

def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=['Adam', 'SGD'],
                        help='Optimizer for training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum of optimizer.')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='Initial learning rate.')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay.')
    
    parser.add_argument('--epochs', type=int,  default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--eval_freq', type=int,  default=1,
                        help='Number of epochs to train.')
    parser.add_argument('--useunlabel', type=str,  default='yes', choices=['yes', 'no'],
                        help='train cr loss with all data or unlabeled data')
    
    parser.add_argument('--lg_k', type=int, default=10,
                        help='k hop for local graph')

    parser.add_argument("--alpha", type=float, default=0.1,
                        help='[0,1] coefficient for feature propagation.')    

    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument("--ce_b", type=float, default=0.5,
                        help='regularize local and global ce') 
    
    parser.add_argument('--cr_loss', type=str,  default='kl', choices=['ce', 'kl', 'l2'],
                        help='kl or l2 for cr loss, select kl or l2 for consis plus')
    parser.add_argument("--lambda_cr", type=float, default=1,
                        help='loss weight of consistency regularization loss')

    parser.add_argument('--max_n', type=int, default=0,
                        help='0 for default using all nodes, otherwise from input')
    parser.add_argument('--zero', type=str,  default='yes', choices=['yes', 'no'],
                        help='make negative cosine similarity to zero?')
    parser.add_argument('--ncr_loss', type=str,  default='weighted_kl', choices=['kl', 'weighted_kl'],
                        help='how to measure ncr loss')
    parser.add_argument("--ncr_conf", type=float, default=0.0,
                        help='conf for ncr loss')
    parser.add_argument("--lambda_ncr", type=float, default=0.8,
                        help='loss weight of neighbor consistency regularization loss')
    parser.add_argument("--ncr_t", type=float, default=0.9,
                        help='tem for consistency regularization loss')
    
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--noisy_rate', type=int,  default=0.1,
                        help='max number of neighbors for local sim.')

    parser.add_argument('--dataset', type=str, default="cora_ml",
                        choices=['citeseer','pubmed', 'cs', 'arxiv', 'cora_ml'], help='dataset')
    parser.add_argument('--ptb_rate', type=float, default=0.4,
                        help="noise ptb_rate")
    parser.add_argument("--label_rate", type=float, default=0.05,
                        help='rate of labeled data')
    parser.add_argument("--val_rate", type=float, default=0.15,
                        help='rate of labeled data')
    parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                        help='type of noises')

    parser.add_argument('--verbose', choices=["True", "False"], default=False,
                        help='printing logs?')
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument("--beta", type=float, default=0.5,
                        help='[0,1] trade-off')

    parser.add_argument('--patience', type=int, default=100)

    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--outf', type=str, default='ours.log')
    args = parser.parse_args()
    
    return args