// ios_coreml_stub.swift
// Minimal CoreML inference for SudokuCell.mlmodel (3 outputs).

import CoreML
import UIKit

class SudokuCellCoreML {
  private let model: MLModel

  init?(url: URL) {
    do { self.model = try MLModel(contentsOf: url) } catch { return nil }
  }

  func predict(cellImage: UIImage) -> (type:[Double], digit:[Double], notes:[Double])? {
    guard let resized = SudokuCellCoreML.resize(image: cellImage, to: CGSize(width:64, height:64)),
          let buf = SudokuCellCoreML.pixelBuffer(from: resized) else { return nil }
    let input = MLDictionaryFeatureProvider(dictionary: ["input_1": buf])
    guard let out = try? model.prediction(from: input) else { return nil }
    // Replace key names according to exported model
    let type = out.featureValue(for: "type")!.multiArrayValue!
    let digit = out.featureValue(for: "digit")!.multiArrayValue!
    let notes = out.featureValue(for: "notes")!.multiArrayValue!
    return (type.array, digit.array, notes.array)
  }

  static func resize(image: UIImage, to size: CGSize) -> UIImage? {
    UIGraphicsBeginImageContextWithOptions(size, true, 1.0)
    image.draw(in: CGRect(origin: .zero, size: size))
    let out = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    return out
  }

  static func pixelBuffer(from image: UIImage) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: true,
                 kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, 64, 64, kCVPixelFormatType_OneComponent8, attrs, &pb)
    guard let px = pb, let cg = image.cgImage else { return nil }
    CVPixelBufferLockBaseAddress(px, .readOnly)
    let ctx = CGContext(data: CVPixelBufferGetBaseAddress(px), width: 64, height: 64, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(px), space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
    ctx.draw(cg, in: CGRect(x:0,y:0,width:64,height:64))
    CVPixelBufferUnlockBaseAddress(px, .readOnly)
    return px
  }
}

extension MLMultiArray {
  var array: [Double] { (0..<count).map { self[$0].doubleValue } }
}
