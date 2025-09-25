// ios_tflite_stub.swift
// Tiny example using TensorFlowLiteSwift to run sudoku_cell_model_int8.tflite
// Podfile: pod 'TensorFlowLiteSwift', '~> 2.14.0'

import Foundation
import TensorFlowLite
import UIKit

final class SudokuCellModelTFLite {
  private var interpreter: Interpreter

  init?(modelPath: String) {
    do {
      var options = Interpreter.Options()
      options.threadCount = 2
      self.interpreter = try Interpreter(modelPath: modelPath, options: options)
      try interpreter.allocateTensors()
    } catch {
      print("Interpreter creation failed: \(error)")
      return nil
    }
  }

  func predict(cellImage: UIImage) -> (type:[Float], digit:[Float], notes:[Float])? {
    guard let input = SudokuCellModelTFLite.preprocess(img: cellImage) else { return nil }
    do {
      try interpreter.copy(input, toInputAt: 0)
      try interpreter.invoke()
      let outType = try interpreter.output(at: 0)
      let outDigit = try interpreter.output(at: 1)
      let outNotes = try interpreter.output(at: 2)
      let typeScores = [Float](unsafeData: outType.data) ?? []
      let digitScores = [Float](unsafeData: outDigit.data) ?? []
      let notesScores = [Float](unsafeData: outNotes.data) ?? []
      return (typeScores, digitScores, notesScores)
    } catch {
      print("Inference failed: \(error)")
      return nil
    }
  }

  static func preprocess(img: UIImage) -> Data? {
    // Resize to 64x64 grayscale, normalize to [0,255] uint8
    UIGraphicsBeginImageContextWithOptions(CGSize(width:64, height:64), true, 1.0)
    img.draw(in: CGRect(x:0,y:0,width:64,height:64))
    let resized = UIGraphicsGetImageFromCurrentImageContext()
    UIGraphicsEndImageContext()
    guard let cg = resized?.cgImage else { return nil }
    let w = cg.width, h = cg.height
    var buf = [UInt8](repeating: 0, count: w*h)
    let cs = CGColorSpaceCreateDeviceGray()
    let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: cs, bitmapInfo: 0)
    ctx?.draw(cg, in: CGRect(x:0,y:0,width:w,height:h))
    // [1,64,64,1] NHWC
    return Data(bytes: buf, count: buf.count)
  }
}

extension Array {
  init?<T>(unsafeData: Data) {
    self = unsafeData.withUnsafeBytes {
      Array<T>($0.bindMemory(to: T.self))
    } as? [Element] ?? []
  }
}
