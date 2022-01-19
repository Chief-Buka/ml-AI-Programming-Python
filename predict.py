import argparse
import MyModel

parser = argparse.ArgumentParser(description="Process arguments for predict using prediction script")

parser.add_argument("image_path", help="specify the path/to/image")
parser.add_argument("checkpoint", help="specify the path to the saved classifier model")
parser.add_argument("--top_k", default=3, help="specify the cutoff for the top predicted classes")
parser.add_argument("--category_names", default='cat_to_name.json', help="specify the mapping of categories to real names")
parser.add_argument("--gpu", help="specify if a gpu should be used for inference", default=False)

args = parser.parse_args()
print(args)

#load the model with specified arguments
model, device = MyModel.load_model(args.checkpoint, args.gpu)

#make prediction with specified arguments
probs, classes = MyModel.predict(args.image_path, model, device, int(args.top_k), args.category_names)

for prob, flower in zip(probs.cpu().numpy().squeeze(), classes):
    print(prob, flower)