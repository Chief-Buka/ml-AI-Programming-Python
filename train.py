import argparse
import MyModel

parser = argparse.ArgumentParser(description="specify arguments for training the model")

parser.add_argument("data_dir", help="specify the directory that contains the data")
parser.add_argument("--save_dir", help="specify the directory in which the model checkpoint should be saved")
parser.add_argument("--arch", default="vgg11", help="specify the architecture to be used for the model")
parser.add_argument("--learning_rate", default=0.01)
parser.add_argument("--hidden_units", default=512)
parser.add_argument("--epochs", default=20)
parser.add_argument("--gpu", default=False)

args = parser.parse_args()
print(args)


######### TRAINING ##########

# get the data loaders for training and validation
dataloaders, class_to_idx = MyModel.get_dataloaders(args.data_dir)

# get the model with the specified architecture
model = MyModel.get_model(args.arch)

# change the classifier part of the model
model = MyModel.change_classifier(model, int(args.hidden_units))

# train the model with specified arguments
MyModel.train(model, dataloaders, float(args.learning_rate), int(args.epochs), args.gpu)

MyModel.save_checkpoint(model, args.save_dir, class_to_idx, args.arch, int(args.hidden_units))
