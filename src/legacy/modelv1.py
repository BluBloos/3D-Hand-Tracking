# NOTE(Noah): This file is legacy code and not something that is currently in use.

def GenerateResidualModule():
    return [ Conv2D(16, 1, activation='relu', padding="same"), Conv2D(16, 3, activation='relu', padding="same"), Conv2D(32, 1, activation='relu', padding="same") ]

def GenerateHourglass():
    return [
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
        GenerateResidualModule(), 
    ]

class MyModel(Model):
    def __init__(self, modelName):
        super(MyModel, self).__init__()
        self._layers = [
            Conv2D(32, 7, activation='relu', input_shape=(320,320, 1), data_format="channels_last", padding="same"),
            GenerateResidualModule(),
            GenerateResidualModule(),
            GenerateHourglass(),
            GenerateResidualModule(),
            GenerateResidualModule(),
            Conv2D(21, 1, activation = 'relu', padding="same"), # 21 keypoints
            Conv2D(1, 1, padding="same"), # turn into probability density thing.
            MaxPool2D(),
            UpSampling2D()
        ]
        self.modelName = modelName
        
    def ResidualModuleForward(self, module, x, training=False):
        out = module[0](x)
        out = module[1](out)
        out = module[2](out)
        return out + x # This is the residual part lol. The skip connection.

    def HourglassForward(self, hourglass, x, training=False):
        maxpool = self._layers[8]
        upsample = self._layers[9]
        out = self.ResidualModuleForward(hourglass[0], x, training)
        skip1 = self.ResidualModuleForward(hourglass[1],out, training) # skip layer
        out = maxpool(out)
        out = self.ResidualModuleForward(hourglass[2],out, training)
        skip2 = self.ResidualModuleForward(hourglass[3],out, training) # skip layer
        out = maxpool(out)
        out = self.ResidualModuleForward(hourglass[4],out, training)
        skip3 = self.ResidualModuleForward(hourglass[5],out, training) # skip layer
        out = maxpool(out)
        out = self.ResidualModuleForward(hourglass[6],out, training)
        skip4 = self.ResidualModuleForward(hourglass[7],out, training) # skip layer
        out = maxpool(out)
        out = self.ResidualModuleForward(hourglass[8],out, training) # begin of 3 residual modules at small res.
        out = self.ResidualModuleForward(hourglass[9],out, training)
        out = self.ResidualModuleForward(hourglass[10],out, training)
        out = upsample(out) + skip4
        out = self.ResidualModuleForward(hourglass[11],out, training)
        out = upsample(out) + skip3
        out = self.ResidualModuleForward(hourglass[12],out, training)
        out = upsample(out) + skip2
        out = self.ResidualModuleForward(hourglass[13],out, training)
        out = upsample(out) + skip1
        out = self.ResidualModuleForward(hourglass[14],out, training)
        return out

    def call(self, x, training=False):
        x = self._layers[0](x)
        x = self.ResidualModuleForward(self._layers[1], x, training)
        skip = self.ResidualModuleForward(self._layers[2], x, training)
        x = self.HourglassForward(self._layers[3], skip, training)
        x = self.ResidualModuleForward(self._layers[4], x, training)
        x = self.ResidualModuleForward(self._layers[5], x, training)
        x = x + skip
        x = self._layers[6](x)
        return self._layers[7](x)

    def summary(self):
        print("Model: ", self.modelName)

model = MyModel("Aristotle")

model.summary() # print out the model in a nice, clean format.