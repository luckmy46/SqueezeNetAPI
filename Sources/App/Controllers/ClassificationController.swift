import Vapor

struct ClassificationController: RouteCollection {
    let imageClassifier = ImageClassifier()

    func boot(routes: RoutesBuilder) throws {
        routes.on(.POST,
                  "classify",
                  body: .collect(maxSize: ByteCount(value: 2000*1024)),
                  use: classification
        )
    }

    func classification(req: Request) throws -> EventLoopFuture<[ClassificationResult]> {
        let eventLoop = req.eventLoop.makePromise(of: [ClassificationResult].self)

        let request = try req.content.decode(ClassifyRequest.self)
        let image = request.image
        let data = Data(buffer: image.data)
        guard let orientation = CGImagePropertyOrientation(rawValue: 0) else { throw Abort(.internalServerError) }

        imageClassifier.getClassification(forImageData: data, orientation: orientation) { result in
            switch result {
            case .success(let classificationResults):
                eventLoop.succeed(classificationResults)
            case .failure(let error):
                eventLoop.fail(error)
            }
        }

        return eventLoop.futureResult
    }
}
