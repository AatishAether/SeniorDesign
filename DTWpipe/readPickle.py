##Take a pickle file as input and display its contents
import pickle
import sys

def main():

    pickleFile = './data/dataset/U/U-36/rh_U-36.pickle' 
    print("Reading data from pickle file",pickleFile)
    with open(pickleFile, 'rb') as f:
        data = pickle.load(f)
    f.close()
    print("Data read from pickle file",pickleFile)
    print(data)

if __name__ == '__main__':
    main()