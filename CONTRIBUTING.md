# Contribution Guide

Thank you for your interest in contributing to Core ML Stable Diffusion! This project was released for system demonstration purposes and there are limited plans for future development of the repository. While we welcome new pull requests and issues please note that our response may be limited.


## Submitting a Pull Request

The project is licensed under the MIT license. By submitting a pull request, you represent that you have the right to license your contribution to Apple and the community, and agree by submitting the patch that your contributions are licensed under the MIT license.

## Code of Conduct

We ask that all community members read and observe our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Note about this fork

`thrtysxty/ml-stable-diffusion` is a community fork maintained by [thrtysxty](https://github.com/thrtysxty) to add inpainting support to the upstream `apple/ml-stable-diffusion` Swift package. The inpainting additions live entirely in the `swift/StableDiffusion/pipeline/` directory and are **purely additive** at the public API level — existing `textToImage` and `imageToImage` callers compile unchanged.

This fork is **not regularly synced with upstream**. The upstream `apple/ml-stable-diffusion` repository has been in maintenance mode since 2024; the last meaningful release predates this fork by more than a year. We track upstream issues for security fixes only and cherry-pick them as needed; do not expect a fast-forward merge.

If you have questions about the inpainting additions specifically, open an issue on `thrtysxty/ml-stable-diffusion`. For everything else, please direct feedback to upstream.
