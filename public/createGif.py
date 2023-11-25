import sys
import json
import base64
import numpy as np
import os
from PIL import Image
import io

def main():
    im = []
    image_array = np.load('uploads/data.npy')
    for i in range(len(image_array)):
        im.append(Image.fromarray(image_array[i].astype('uint8')))

    # duration is the number of milliseconds between frames; this is 40 frames per second
    im[0].save("public/scan.gif", save_all=True, append_images=im[1:], duration=50, loop=0)
    print("healthy")



def main2():

    print(type(sys.argv[1]))
    print(type('.argv[1]'))


    
    # print(gifImage)
    # obj = json.loads(sys.argv[1])
    datapath = str(os.path.dirname(os.path.realpath(__file__)))+'/file.txt'
    f = open(datapath,'r')

    base64data = f.read()
    # print(base64data)

    npArry = base64.b64decode(base64data)

    print(len(npArry))
    # image = Image.open(io.BytesIO(npArry[0]))
    # image_np = np.array(image)
    # print(image_np)


    


    # base64_decoded = base64.b64decode(test_image_base64_encoded)
    
    
    
    # image_torch = torch.tensor(np.array(image))

    # vector_bytes = vector_np.tobytes()
    # vector_bytes_str = str(vector_bytes)
    # vector_bytes_str_enc = vector_bytes_str.encode()




    # bytes_np_dec = npArry.decode('unicode-escape').encode('ISO-8859-1')[2:-1]
    # np.frombuffer(bytes_np_dec, dtype=np.float64)




    # nparr = np.fromstring(base64.b64decode(base64data), np.uint8)

    # q = np.frombuffer(npArry)

    # print("len", nparr)
    # print(os.path.dirname(os.path.realpath(__file__)))

if __name__ == '__main__':
    main()
