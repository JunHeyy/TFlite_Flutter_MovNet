import 'dart:io';
import 'dart:isolate';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as imageLib;
import 'package:object_detection/tflite/classifier.dart';
import 'package:object_detection/tflite/stats.dart';
import 'package:object_detection/utils/image_utils.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../tflite/recognition.dart';

/// Manages separate Isolate instance for inference
class IsolateUtils {
  static const String DEBUG_NAME = "InferenceIsolate";

  late Isolate _isolate;
  ReceivePort _receivePort = ReceivePort();
  late SendPort _sendPort;

  SendPort get sendPort => _sendPort;

  Future<void> start() async {
    _isolate = await Isolate.spawn<SendPort>(
      entryPoint,
      _receivePort.sendPort,
      debugName: DEBUG_NAME,
    );

    _sendPort = await _receivePort.first;
  }

  static void entryPoint(SendPort sendPort) async {
    final port = ReceivePort();
    sendPort.send(port.sendPort);
    GpuDelegate gpuDelegate = GpuDelegate(
        options: GpuDelegateOptions(
            allowPrecisionLoss: true, waitType: TFLGpuDelegateWaitType.active));

    // Buffer
    InterpreterOptions interpreterOptions = InterpreterOptions()
      ..addDelegate(gpuDelegate)
      ..threads = 6;
    // final buffer = await getBuffer('assets/your_model.tflite');

    await for (final IsolateData isolateData in port) {
      if (isolateData != null) {
        Classifier classifier = Classifier(
            interpreter:
                Interpreter.fromAddress(isolateData.interpreterAddress));

        imageLib.Image? image =
            ImageUtils.convertCameraImage(isolateData.cameraImage);
        if (Platform.isAndroid) {
          image = imageLib.copyRotate(image!, 90);
        }
        // Map<String, dynamic>? results = classifier.predict(image!);
        // KeypointRecognition? results2 = classifier.predictMovNet(image!);
        classifier.predictMovNet(image!);

        Map<String, dynamic> dummyResult = {
          "recognitions": [
            Recognition(0, "toilet", 0.75, Rect.fromLTRB(0.1, 0.1, 0.2, 0.2)),
            Recognition(1, "toilet", 0.50, Rect.fromLTRB(0.3, 0.3, 0.4, 0.4)),
          ],
          "stats": Stats(
              totalPredictTime: 1000,
              inferenceTime: 500,
              preProcessingTime: 200,
              totalElapsedTime: 10)
        };

        isolateData.responsePort.send(dummyResult);
      }
    }
  }
}

/// Bundles data to pass between Isolate
class IsolateData {
  CameraImage cameraImage;
  int interpreterAddress;

  late SendPort responsePort;

  IsolateData(
    this.cameraImage,
    this.interpreterAddress,
  );
}

Future<Uint8List> getBuffer(String filePath) async {
  final rawAssetFile = await rootBundle.load(filePath);
  final rawBytes = rawAssetFile.buffer.asUint8List();
  return rawBytes;
}
