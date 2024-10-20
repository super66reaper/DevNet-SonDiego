import LRC_Neural_Network_Model as nn

# DATA SETS:
# https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification
# https://www.kaggle.com/datasets/prasanshasatpathy/leaves-healthy-or-diseased
# https://www.kaggle.com/datasets/amandam1/healthy-vs-diseased-leaf-image-dataset
# Actual Data for Tom Diago
# https://drive.google.com/drive/folders/1u8ZzB1cB5ClDCD7g6sNikSLksbGr9j4f

def createModel(showModel) :
    # Create a base model from the framework
    model = nn.Model()

    # Here is the main meat
    # Adding the layers to the model (Each layer has an activation after it, softmax is good for final output layer, rest can be RELU)
    model.add(nn.Layer_Dense(4, 16))
    model.add(nn.Activation_ReLU())
    model.add(nn.Layer_Dense(16, 3))
    model.add(nn.Activation_Softmax())

    if (showModel):
        model.printModelImage()

    model.set(
        loss=nn.Loss_CategoricalCrossentropy(),
        optimizer=nn.Optimizer_Adam(decay=0),
        accuracy=nn.Accuracy_Categorical()
    )

    model.finalize()

    return model

model = createModel(True)

model.train(X=np.array([data1]), y=np.array([yTruth1]), epochs=1)