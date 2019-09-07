import argparse
from .utils.split import split
from .utils.move_non import move
from .predict import predict
from .utils.combine import combine
import tempfile





def create_temp_dir():
    split_path = tempfile.mkdtemp(suffix='_split', prefix='skyunet_')
    combine_path = tempfile.mkdtemp(suffix='_combine', prefix='skyunet_')
    print(split_path)
    print(combine_path)
    return str(split_path), str(combine_path)


def execute(image, output, model='/tmp/weights.hdf5', batch_size=5, threshold=.7):
    split_path, combine_path = create_temp_dir()
    print('split image to tiles...')
    split(image, split_path)
    move(split_path, combine_path)
    predict(split_path, combine_path, model, batch_size=batch_size, threshold=threshold)
    print('creating output image...')
    combine(combine_path, output)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Predict Image')
    parser.add_argument('image', type=str, help='Input image')
    parser.add_argument('output', type=str, help='Output image')
    parser.add_argument('--model', '-m', type=str, default='/tmp/trained_weights.hdf5', help="Model file, for example: /tmp/weights.hdf5")
    parser.add_argument('--batch_size', '-b', type=int, default=5, help="batch size")
    parser.add_argument('--threshold', '-t', type=float, default=0.7, help="threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    execute(args.image, args.output, model=args.model, batch_size=args.batch_size, threshold=args.threshold)

if __name__=='__main__':
    main()
