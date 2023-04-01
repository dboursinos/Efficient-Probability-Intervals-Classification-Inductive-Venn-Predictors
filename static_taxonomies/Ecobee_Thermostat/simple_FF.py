from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation

def create_base_model(in_dims,embeddings,num_classes):
       """
       Base network to be shared.
       """
       model = Sequential()
       model.add(Dense(10, activation='relu', input_dim=in_dims[0]))
       model.add(Dense(units = embeddings,activation='sigmoid',name='embedding'))
       model.add(Dense(units = num_classes,name='logits'))
       model.add(Activation('softmax'))
       return model
