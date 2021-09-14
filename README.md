# SqueezeNet API - A CoreML api
This is a demo project for a CoreML Vapor API. This demo project has one post route called `classify` which takes in a multipart file called image. It will then try to classify this image using Apples Image Classifier model, found here: [https://ml-assets.apple.com/coreml/models/Image/ImageClassification/SqueezeNet/SqueezeNet.mlmodel](https://ml-assets.apple.com/coreml/models/Image/ImageClassification/SqueezeNet/SqueezeNet.mlmodel)

## Tutorial
This is a demo project written by Paul Peelen. For the totorial to this project along with instructions on how to use it, please see the tutorial posted here:
[https://www.PaulPeelen.com/CoreMLAPI](https://paulpeelen.com/CoreMLAPI)

## MLModel
Download the model mentioned above to `Resources/coreml` and follow these instructions from the root of the project to compile it:

```bash
xcrun coremlcompiler compile Resources/coreml/SqueezeNet.mlmodel Resources/coreml
```

Once completed, a set of new files will have been created in the same location and the mlmodel was stored. 
Now that this is done, we need to generate the source code to this interface. Back in the terminal, execute the following command:

```bash
xcrun coremlcompiler generate Resources/coreml/SqueezeNet.mlmodel --language Swift Sources/App/
```

## Requirements
This Vapor project will only run on a MacOS machine.
