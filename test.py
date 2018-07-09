import json
import numpy as np
import h5py
import os

with open('rps_metadata.json') as f:
	data = json.load(f)



sz=len(data)
hf = h5py.File('data.h5', 'w')

#buff = open("rps_weights.buf", "rb")
fdata = np.fromfile("rps_weights.buf", '>f4') 

layername=""
for i in range(0, sz):
  if (data[i]['layer_name'] !=layername):
     g1=hf.create_group(data[i]['layer_name'])
     layername= data[i]['layer_name']
  if (data[i]['type']=='float32'):
    ofs=data[i]['offset']/4
    lens=data[i]['length']
    z=fdata[ofs:ofs+lens]
    shap=data[i]['shape']
    z=np.reshape(z, shap)
    print z.shape
    w=data[i]['weight_name'].split('/')
    
    print w
    if (hf.get(data[i]['layer_name']+"/"+w[0]) is None):
       print "mone"
       g2=hf.create_group(data[i]['layer_name']+"/"+w[0])
    else:
       print "dfd"
       g2=hf.get(data[i]['layer_name']+"/"+w[0])

    g2.create_dataset(w[1], data=z)

hf.close()
