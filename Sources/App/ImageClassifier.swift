import Vapor
import CoreML
import Vision
import CoreImage

enum ImageClassifierError: Error {
    case nothingRecognised
    case unableToClassify
    case invalidImage
}

struct ClassificationResult: Content {
    let identifier: String
    let confidence: Float
}

class ImageClassifier {
    typealias classificationCompletion = ((Result<[ClassificationResult], ImageClassifierError>) -> Void)
    private var completionClosure: classificationCompletion?

    lazy var request: VNCoreMLRequest = {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        do {
            let model = try VNCoreMLModel(for: SqueezeNet(configuration: config).model)
            let request = VNCoreMLRequest(model: model, completionHandler: handleClassifications)
            request.imageCropAndScaleOption = .centerCrop

            return request
        } catch {
            fatalError("Failed to load the model. Error: \(error.localizedDescription)")
        }
    }()

    func getClassification(forImageData imageData: Data, orientation: CGImagePropertyOrientation, completion: @escaping classificationCompletion) {
        guard let ciImage = CIImage(data: imageData) else {
            completion(.failure(.invalidImage))
            return
        }

        completionClosure = completion

        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
        do {
            try handler.perform([self.request])
        } catch {
            print("Failed to perform classification.\n\(error.localizedDescription)")
        }
    }
}

private extension ImageClassifier {
    func handleClassifications(forRequest request: VNRequest, error: Error?) {
        guard
            let results = request.results,
            let classifications = results as? [VNClassificationObservation]
        else {
            debugPrint("Unable to classify image.\n\(error!.localizedDescription)")
            completionClosure?(.failure(.unableToClassify))
            return
        }

        if classifications.isEmpty {
            completionClosure?(.failure(.nothingRecognised))
        } else {
            let topClassifications = classifications.prefix(2)
            let results = topClassifications.map { classification in
                ClassificationResult(identifier: classification.identifier, confidence: classification.confidence)
            }
            completionClosure?(.success(results))
        }
    }
}
