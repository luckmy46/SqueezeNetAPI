import Vapor

struct ClassifyRequest: Content {
    let image: File
}
