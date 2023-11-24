
# plt.imshow(img_array, cmap='gray')
# plt.show()

# this might fail if `img_array` contains a data type that is not supported by PIL,
# in which case you could try casting it to a different dtype e.g.:
# im = Image.fromarray(img_array.astype(np.uint8))



# img_array = np.load('data/0000.npy') # many arrays
# print(img_array)
# im = Image.fromarray(img_array[0])
# im.show()



# aklfjdsaljfdafdsal;fdsakljfsd





#! /usr/bin/env python3
import base64
import io
import numpy as np
from PIL import Image
from viaduc import Viaduc


im = []
image_array = np.load('data/0001.npy')
# print(image_array)



for i in range(len(image_array)):
    im.append(Image.fromarray(image_array[i].astype('uint8')))
    print(image_array[i])

buffer = io.BytesIO()
im[0].save(buffer, format='GIF', save_all=True, append_images=im[1:], optimize=False, duration=200, loop=0)
buffer.seek(0)
data_uri = base64.b64encode(buffer.read()).decode('ascii')



class Presentation(Viaduc.Presentation):
    width = 300
    height = 300
    title = 'gif'
    html = '''
<!DOCTYPE html>
  <head>
    {{bootstrap_meta}} {{bootstrap_css}}
    <title>{{title}}</title>
  </head>
  <body>
    <img src="data:image/gif;base64,''' + data_uri + '''">
    {{bootstrap_js}}
  </body>  
 </html>
'''


if __name__ == '__main__':
    Viaduc(presentation=Presentation())