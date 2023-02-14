import 'dart:math';
import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/tflite/recognition.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'stats.dart';

/// Classifier
class Classifier {
  /// Instance of Interpreter
  late Interpreter _interpreter;

  /// Input size of image (height = width = 300)
  static const int INPUT_SIZE = 256;

  /// Result score threshold
  static const double THRESHOLD = 0.25;

  /// [ImageProcessor] used to pre-process the image
  ImageProcessor? imageProcessor;

  /// Padding the image to transform into square
  late int padSize;

  /// Shapes of output tensors
  late List<List<int>> _outputShapes;

  /// Types of output tensors
  late List<TfLiteType> _outputTypes;

  /// Number of results to show
  static const int NUM_RESULTS = 10;

  Classifier({
    required Interpreter interpreter,
  }) {
    _interpreter = interpreter;
    loadModel(interpreter: interpreter);
  }

  /// Loads interpreter from asset
  void loadModel({required Interpreter interpreter}) async {
    try {
      ///  The following code is Commented out as intepreter is created in camera view instead.
      // GpuDelegate gpuDelegate = GpuDelegate(
      //     options: GpuDelegateOptions(
      //         allowPrecisionLoss: true,
      //         waitType: TFLGpuDelegateWaitType.active));

      // // Buffer
      // InterpreterOptions interpreterOptions = InterpreterOptions()
      //   ..addDelegate(gpuDelegate);
      // _interpreter = interpreter ??
      //     await Interpreter.fromAsset(
      //       MODEL_FILE_NAME,
      //       options: interpreterOptions,
      //     );
      // _interpreter = this.interpreter;

      var outputTensors = _interpreter.getOutputTensors();
      _outputShapes = [];
      _outputTypes = [];
      outputTensors.forEach((tensor) {
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
      });
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  /// Pre-process the image
  TensorImage getProcessedImage(TensorImage inputImage) {
    padSize = max(inputImage.height, inputImage.width);
    if (imageProcessor == null) {
      imageProcessor = ImageProcessorBuilder()
          .add(ResizeWithCropOrPadOp(padSize, padSize))
          .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeMethod.BILINEAR))
          .build();
    }
    inputImage = imageProcessor!.process(inputImage);
    return inputImage;
  }

  void predictMovNet(imageLib.Image image) {
    var predictStartTime = DateTime.now().millisecondsSinceEpoch;

    if (_interpreter == null) {
      print("Interpreter not initialized");
      return null;
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    // Create TensorImage from image
    TensorImage inputImage = TensorImage.fromImage(image);

    // Pre-process TensorImage
    inputImage = getProcessedImage(inputImage);

    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;

    // TensorBuffers for output tensors
    TensorBuffer outputLocations = TensorBufferFloat(_outputShapes[0]);

    // Inputs object for runForMultipleInputs
    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    List<Object> inputs = [inputImage.buffer];

    // Outputs map
    Map<int, Object> outputs = {
      0: outputLocations.buffer,
    };

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    /// Run movenet model
    _interpreter.runForMultipleInputs(inputs, outputs);
    List outputParsed = [];

    /// Code below to get the actual value of x coordinates, y coordinates and confidence scores.
    /// Unpacks the list from the output tensor outputLocations into [x, y and confidence level]
    List<double> data = outputLocations.getDoubleList();
    var x, y, c;
    int TOTAL_ITEMS_IN_SHAPE = 51;
  }

  /// Reference function from code example.
  /// Not used in this project.
  /// Runs object detection on the input image
  Map<String, dynamic>? predict(imageLib.Image image) {
    var predictStartTime = DateTime.now().millisecondsSinceEpoch;

    if (_interpreter == null) {
      print("Interpreter not initialized");
      return null;
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    // Create TensorImage from image
    TensorImage inputImage = TensorImage.fromImage(image);

    // Pre-process TensorImage
    inputImage = getProcessedImage(inputImage);

    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;

    // TensorBuffers for output tensors
    TensorBuffer outputLocations = TensorBufferFloat(_outputShapes[0]);
    TensorBuffer outputClasses = TensorBufferFloat(_outputShapes[1]);
    TensorBuffer outputScores = TensorBufferFloat(_outputShapes[2]);
    TensorBuffer numLocations = TensorBufferFloat(_outputShapes[3]);

    // Inputs object for runForMultipleInputs
    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    List<Object> inputs = [inputImage.buffer];

    // Outputs map
    Map<int, Object> outputs = {
      0: outputLocations.buffer,
      1: outputClasses.buffer,
      2: outputScores.buffer,
      3: numLocations.buffer,
    };

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;

    // run inference
    _interpreter.runForMultipleInputs(inputs, outputs);

    var inferenceTimeElapsed =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;

    // Maximum number of results to show
    int resultsCount = min(NUM_RESULTS, numLocations.getIntValue(0));

    // Using labelOffset = 1 as ??? at index 0
    int labelOffset = 1;

    // Using bounding box utils for easy conversion of tensorbuffer to List<Rect>
    List<Rect> locations = BoundingBoxUtils.convert(
      tensor: outputLocations,
      valueIndex: [1, 0, 3, 2],
      boundingBoxAxis: 2,
      boundingBoxType: BoundingBoxType.BOUNDARIES,
      coordinateType: CoordinateType.RATIO,
      height: INPUT_SIZE,
      width: INPUT_SIZE,
    );

    List<Recognition> recognitions = [];

    for (int i = 0; i < resultsCount; i++) {
      // Prediction score
      var score = outputScores.getDoubleValue(i);

      // Label string
      var labelIndex = outputClasses.getIntValue(i) + labelOffset;
      var label = "toilet";

      if (score > THRESHOLD) {
        // inverse of rect
        // [locations] corresponds to the image size 300 X 300
        // inverseTransformRect transforms it our [inputImage]
        Rect transformedRect = imageProcessor!
            .inverseTransformRect(locations[i], image.height, image.width);

        recognitions.add(
          Recognition(i, label!, score, transformedRect),
        );
      }
    }

    var predictElapsedTime =
        DateTime.now().millisecondsSinceEpoch - predictStartTime;

    // print('Inference time $predictElapsedTime');

    return {
      "recognitions": recognitions,
      "stats": Stats(
          totalPredictTime: predictElapsedTime,
          inferenceTime: inferenceTimeElapsed,
          preProcessingTime: preProcessElapsedTime,
          totalElapsedTime: 10)
    };
  }

  /// Gets the interpreter instance
  Interpreter get interpreter => _interpreter;

  /// Gets the loaded labels
  // List<String> get labels => _labels!;
}
