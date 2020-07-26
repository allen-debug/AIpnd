import argparse

def get_input_args():
    """
    Retrieves and parses the 7 command line arguments provided by the user when
    they run the program from a terminal window. This function. This function returns these arguments as an
    ArgumentParser object.
    """
    
    # Create Parse using ArgumentParser
    ap = argparse.ArgumentParser(
    description='predict-file')
    ap.add_argument('image_path', default='./flowers/test/1/image_06743.jpg',  action="store", type = str)
    ap.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
    ap.add_argument('checkpoint', default='./checkpoint.pth', action="store", type = str)
    ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    
    return ap.parse_args()
