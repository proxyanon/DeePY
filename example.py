from DeePY import Layer, Network
import numpy as np

inputs = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
outputs = np.array([[1], [0], [1]])

l = Layer(True)
l.create('input', sizes=[3, 3], activation='sigmoid')
l.create('hidden', sizes=[3, 5], activation='relu')
l.create('hidden', sizes=[5, 4], activation='sigmoid')
l.create('output', sizes=[4, 1], activation='sigmoid')
layers = l.layers()

dpy = Network(layers=layers, eta=1, verbose=True)
dpy.train(inputs=inputs, outputs=outputs, epochs=1500)
dpy.saveweights()
dpy.savemodel()


ret = dpy.predict(inpt=np.array([0, 0, 0]))
prediction = ret['layer']
rounded = round(prediction[0])

print 'input =>', np.array([0, 0, 0]), 'predict =>', prediction, 'predict rounded =>', rounded
