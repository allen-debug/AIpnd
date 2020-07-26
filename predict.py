import argparse
import json
import torch
import pre_file


def get_command_line_args():
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument('input', type=str,
                        help='Image file')
    
    parser.add_argument('checkpoint', type=str,action='store', default='densenet121',
                        help='Saved model checkpoint')

    #
        
    parser.add_argument('--top_k', type=int,
                        help='Return the top K most likely classes')
    parser.set_defaults(top_k=1)
    
    parser.add_argument('--category_names', type=str,
                        help='File of category names')

    parser.add_argument('--gpu', dest='gpu',
                        action='store_true', help='Use GPU')
    parser.set_defaults(gpu=False)

    return parser.parse_args()


def main():
    
    args = get_command_line_args()
    
    use_gpu = torch.cuda.is_available() and args.gpu
    
    
    print("Input file: {}".format(args.input))
    
    print("Checkpoint file: {}".format(args.checkpoint))
    
    if args.top_k:
        
        print("Returning {} most likely classes".format(args.top_k))
        
    if args.category_names:
        
        print("Category names file: {}".format(args.category_names))
        
    if use_gpu:
        
        print("Using GPU.")
        
    else:
        
        print("Using CPU.")
    
    
    model = pre_file.load_checkpoint(args.checkpoint)
    
    print("Checkpoint loaded.")
    
    
    if use_gpu:
        
        model.cuda()
    
    
    if args.category_names:
        
        with open(args.category_names, 'r') as f:
            
            categories = json.load(f)
            
            print("Category names loaded")
    
    results_to_show = args.top_k if args.top_k else 1
    
    
    
    print("Processing image")
    
    probabilities, clss = pre_file.predict(args.input, model, use_gpu, results_to_show, args.top_k)
    
    
    if results_to_show > 1:
        
        print("Top {} Classes for '{}':".format(len(clss), args.input))
        

        if args.category_names:
            
            print("{:<30} {}".format("Flower", "Probability"))
            
            print("----------------------------------------")
            
        else:
            print("{:<10} {}".format("Class", "Probability"))
            
            print("--------------------")

        for i in range(0, len(clss)):
            
            if args.category_names:
                
                print("{:<30} {:.2f}".format(categories[clss[i]], probabilities[i]))
                
            else:
                
                print("{:<10} {:.2f}".format(clss[i], probabilities[i]))
                
    else:
        
        print("The most likely class is '{}': probability: {:.2f}" \
              
              .format(categories[clss[0]] if args.category_names else clss[0], probabilities[0]))
        
    
if __name__ == "__main__":
    main()
