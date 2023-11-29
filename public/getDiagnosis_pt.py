# import torch
import sys




def main():
    print(sys.argv[1]) # should be the plane

    model_axial_acl = "models/cnn_axial_acl_02.pt"
    model_axial_meniscus = "models/cnn_axial_meniscus_03.pt"




    # model = torch.jit.load('model_scripted.pt')
    # model.eval()

if __name__ == '__main__':
    main()