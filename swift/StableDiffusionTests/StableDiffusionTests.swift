// For licensing see accompanying LICENSE.md file.
// Copyright (C) 2022 Apple Inc. All Rights Reserved.

import XCTest
import CoreML
@testable import StableDiffusion

@available(iOS 16.2, macOS 13.1, *)
final class StableDiffusionTests: XCTestCase {

    var vocabFileInBundleURL: URL {
        let fileName = "vocab"
        guard let url = Bundle.module.url(forResource: fileName, withExtension: "json") else {
            fatalError("BPE tokenizer vocabulary file is missing from bundle")
        }
        return url
    }

    var mergesFileInBundleURL: URL {
        let fileName = "merges"
        guard let url = Bundle.module.url(forResource: fileName, withExtension: "txt") else {
            fatalError("BPE tokenizer merges file is missing from bundle")
        }
        return url
    }

    func testBPETokenizer() throws {

        let tokenizer = try BPETokenizer(mergesAt: mergesFileInBundleURL, vocabularyAt: vocabFileInBundleURL)

        func testPrompt(prompt: String, expectedIds: [Int]) {

            let (tokens, ids) = tokenizer.tokenize(input: prompt)

            print("Tokens          = \(tokens)\n")
            print("Expected tokens = \(expectedIds.map({ tokenizer.token(id: $0) }))")
            print("ids             = \(ids)\n")
            print("Expected Ids    = \(expectedIds)\n")

            XCTAssertEqual(ids,expectedIds)
        }

        testPrompt(prompt: "a photo of an astronaut riding a horse on mars",
                   expectedIds: [49406, 320, 1125, 539, 550, 18376, 6765, 320, 4558, 525, 7496, 49407])

        testPrompt(prompt: "Apple CoreML developer tools on a Macbook Air are fast",
                   expectedIds: [49406,  3055, 19622,  5780, 10929,  5771,   525,   320, 20617,
                                 1922,   631,  1953, 49407])
    }

    func test_randomNormalValues_matchNumPyRandom() {
        var random = NumPyRandomSource(seed: 12345)
        let samples = random.normalArray(count: 10_000)
        let last5 = samples.suffix(5)

        // numpy.random.seed(12345); print(numpy.random.randn(10000)[-5:])
        let expected = [-0.86285345, 2.15229409, -0.00670556, -1.21472309, 0.65498866]

        for (value, expected) in zip(last5, expected) {
            XCTAssertEqual(value, expected, accuracy: .ulpOfOne.squareRoot())
        }
    }

    /// Smoke test for the inpainting additions on `PipelineConfiguration`:
    /// the new `mask` and `maskedImage` fields exist, the computed `mode`
    /// returns `.inPainting` when both are set, the inpainting case is
    /// mutually exclusive with `.textToImage` and `.imageToImage`, and the
    /// configuration remains `Hashable` (existing contract preserved).
    func testInpaintingConfigurationMode() throws {
        // Use any CGImage as a stand-in for the mask and masked image; the
        // computed `mode` only inspects pointer presence.
        let dummyImage = try dummyGrayCGImage(width: 8, height: 8, value: 0xFF)

        var config = PipelineConfiguration(prompt: "a cat in a garden")
        XCTAssertNil(config.mask)
        XCTAssertNil(config.maskedImage)
        XCTAssertEqual(config.mode, .textToImage)

        config.mask = dummyImage
        XCTAssertEqual(config.mode, .textToImage,
                       "mask alone is not enough to enter inpainting mode")

        config.maskedImage = dummyImage
        XCTAssertEqual(config.mode, .inPainting)

        // Drop only one input and the mode must fall back — never inpainting
        // without both inputs, never image-to-image unless startingImage + strength are set.
        config.mask = nil
        XCTAssertEqual(config.mode, .textToImage)
        config.mask = dummyImage

        // Hashable: equal configs (same prompt, same mask + maskedImage) compare equal.
        var twin = PipelineConfiguration(prompt: "a cat in a garden")
        twin.mask = dummyImage
        twin.maskedImage = dummyImage
        XCTAssertEqual(config, twin)
        XCTAssertEqual(config.hashValue, twin.hashValue)
    }

    /// Smoke test for `CGImage.planarMaskShapedArray`: downsample a 64x64 mask
    /// of all-ones to a 8x8 latent-sized buffer and confirm the resulting
    /// array has the expected shape and content.
    func testPlanarMaskShapedArrayDownsample() throws {
        let mask = try dummyGrayCGImage(width: 64, height: 64, value: 0xFF)
        let downsampled = try mask.planarMaskShapedArray(targetHeight: 8, targetWidth: 8)
        XCTAssertEqual(downsampled.shape, [1, 1, 8, 8])
        let scalars = downsampled.scalars
        for v in scalars {
            XCTAssertEqual(v, 1.0, accuracy: 0.01,
                           "all-white source mask should produce ~1.0 entries after downsample")
        }
    }

    // MARK: - Test helpers

    /// Build a solid-gray CGImage suitable as a stand-in mask or masked image
    /// in configuration tests. Avoids loading any on-disk resource.
    private func dummyGrayCGImage(width: Int, height: Int, value: UInt8) throws -> CGImage {
        let bytesPerRow = width
        let capacity = width * height
        var data = [UInt8](repeating: value, count: capacity)
        let colorSpace = CGColorSpace(name: CGColorSpace.linearGray)!
        let bitmapInfo = CGImageAlphaInfo.none.rawValue
        let context = data.withUnsafeMutableBufferPointer { ptr -> CGContext? in
            CGContext(
                data: ptr.baseAddress,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            )
        }
        guard let ctx = context,
              let image = ctx.makeImage() else {
            throw NSError(domain: "StableDiffusionTests", code: 1)
        }
        return image
    }
}
