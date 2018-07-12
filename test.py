import json
import numpy as np
import h5py
import os
from array import array

with open('rps_metadata.json') as f:
    data = json.load(f)

sz=len(data)
hf = h5py.File('data_95.h5', 'w')

#buff = open("rps9_weights.buf", "rb")
#fdata = np.fromfile("rps_weights.buf", '<f4') #should be little edian (<f4), 
fdata = np.fromfile("rps-0.9575_weights.buf", '<f4')
# big edian won't work >f4 

layername=""
for i in range(0, sz):
  if (data[i]['layer_name'] !=layername):
     g1=hf.create_group(data[i]['layer_name'])
     layername= data[i]['layer_name']
     wm=[]
     wm.append(data[i]['weight_name'])
  else:
     wm.append(data[i]['weight_name'])

  if (data[i]['type']=='float32'):
    ofs=data[i]['offset']/4
    lens=data[i]['length']
    z=fdata[ofs:ofs+lens]
    shap=data[i]['shape']
    z=np.reshape(z, shap)
    print z.shape
    w=data[i]['weight_name'].split('/')
    
    if (hf.get(data[i]['layer_name']+"/"+w[0]) is None):
       g2=hf.create_group(data[i]['layer_name']+"/"+w[0])
    else:
       g2=hf.get(data[i]['layer_name']+"/"+w[0])

    g2.create_dataset(w[1], data=z)
    
    g1.attrs['weight_name']=np.array(wm, dtype='S')

with open('rps.json') as f:
    ndata = json.load(f)

hf.attrs['backend']=ndata['backend']
hf.attrs['keras_version']=ndata['keras_version']
layers=ndata['config']['layers']

sz=len(layers)
lay=[];
for i in range(0, sz):
  if (hf.get(layers[i]['name']) is None):
      g2=hf.create_group(layers[i]['name'])
      g2.attrs['weight_name']=np.array([], dtype=np.float64)


#writing array into attributes
  lay.append(layers[i]['name'])
  print layers[i]['name']

hf.attrs['layer_names']=np.array(lay, dtype='S')


hf.close()
