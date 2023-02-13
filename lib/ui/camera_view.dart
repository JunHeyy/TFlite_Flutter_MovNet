import 'dart:io';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:object_detection/tflite/classifier.dart';
import 'package:object_detection/tflite/recognition.dart';
import 'package:object_detection/tflite/stats.dart';
import 'package:object_detection/ui/camera_view_singleton.dart';
import 'package:object_detection/utils/isolate_utils.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

/// [CameraView] sends each frame for inference
class CameraView extends StatefulWidget {
  /// Callback to pass results after inference to [HomeView]
  final Function(List<Recognition> recognitions) resultsCallback;

  /// Callback to inference stats to [HomeView]
  // final Function(Stats stats) statsCallback;

  /// Constructor
  const CameraView(this.resultsCallback);
  @override
  _CameraViewState createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> with WidgetsBindingObserver {
  /// List of available cameras
  late List<CameraDescription> cameras;

  /// Controller
  CameraController? cameraController;

  /// true when inference is ongoing
  late bool predicting;

  /// Instance of [Classifier]
  Classifier? classifier;

  late Interpreter interpreter;

  /// Instance of [IsolateUtils]
  late IsolateUtils isolateUtils;

  @override
  void initState() {
    super.initState();
    initStateAsync();
  }

  void initStateAsync() async {
    WidgetsBinding.instance.addObserver(this);

    // Create an instance of classifier to load model and labels
    final dataFile = await getFile('assets/detect.tflite');
    if (Platform.isIOS) {
      GpuDelegate gpuDelegate = GpuDelegate(
          options: GpuDelegateOptions(
              allowPrecisionLoss: false,
              waitType: TFLGpuDelegateWaitType.doNotWait,
              enableQuantization: false));

      // Buffer
      InterpreterOptions interpreterOptions = InterpreterOptions()
        ..addDelegate(gpuDelegate)
        ..threads = 6;
      interpreter = Interpreter.fromFile(dataFile, options: interpreterOptions);
      classifier = Classifier(interpreter: interpreter);
    } else {
      interpreter = Interpreter.fromFile(dataFile,
          options: InterpreterOptions()..threads = 6);
    }

    // Spawn a new isolate
    isolateUtils = IsolateUtils();
    await isolateUtils.start();

    // Camera initialization
    initializeCamera();

    // Initially predicting = false
    predicting = false;
  }

  /// Initializes the camera by setting [cameraController]
  void initializeCamera() async {
    cameras = await availableCameras();

    // cameras[0] for rear-camera
    cameraController =
        CameraController(cameras[0], ResolutionPreset.low, enableAudio: false);

    cameraController?.initialize().then((_) async {
      // Stream of image passed to [onLatestImageAvailable] callback
      await cameraController?.startImageStream(onLatestImageAvailable);

      /// previewSize is size of each image frame captured by controller
      ///
      /// 352x288 on iOS, 240p (320x240) on Android with ResolutionPreset.low
      Size? previewSize = cameraController?.value.previewSize;

      /// previewSize is size of raw input image to the model
      CameraViewSingleton.inputImageSize = previewSize!;

      // the display width of image on screen is
      // same as screenWidth while maintaining the aspectRatio
      Size screenSize = MediaQuery.of(context).size;
      CameraViewSingleton.screenSize = screenSize;
      CameraViewSingleton.ratio = screenSize.width / previewSize.height;
    });
  }

  @override
  Widget build(BuildContext context) {
    // Return empty container while the camera is not initialized
    if (cameraController == null || !cameraController!.value.isInitialized) {
      return Container();
    }

    return AspectRatio(
        aspectRatio: cameraController!.value.aspectRatio,
        child: CameraPreview(cameraController!));
  }

  /// Callback to receive each frame [CameraImage] perform inference on it
  onLatestImageAvailable(CameraImage cameraImage) async {
    if (interpreter != null) {
      // If previous inference has not completed then return
      if (predicting) {
        return;
      }

      setState(() {
        predicting = true;
      });

      var uiThreadTimeStart = DateTime.now().millisecondsSinceEpoch;

      // Data to be passed to inference isolate
      var isolateData = IsolateData(cameraImage, interpreter.address);

      // We could have simply used the compute method as well however
      // it would be as in-efficient as we need to continuously passing data
      // to another isolate.

      /// perform inference in separate isolate
      Map<String, dynamic> inferenceResults = await inference(isolateData);

      var uiThreadInferenceElapsedTime =
          DateTime.now().millisecondsSinceEpoch - uiThreadTimeStart;

      // pass results to HomeView
      widget.resultsCallback(inferenceResults["recognitions"]);

      // pass stats to HomeView
      // widget.statsCallback((inferenceResults["stats"] as Stats)
      //   ..totalElapsedTime = uiThreadInferenceElapsedTime);

      // set predicting to false to allow new frames
      setState(() {
        predicting = false;
      });
    }
  }

  /// Runs inference in another isolate
  Future<Map<String, dynamic>> inference(IsolateData isolateData) async {
    ReceivePort responsePort = ReceivePort();
    isolateUtils.sendPort
        .send(isolateData..responsePort = responsePort.sendPort);
    var results = await responsePort.first;
    return results;
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) async {
    switch (state) {
      case AppLifecycleState.paused:
        cameraController!.stopImageStream();
        break;
      case AppLifecycleState.resumed:
        if (!cameraController!.value.isStreamingImages) {
          await cameraController!.startImageStream(onLatestImageAvailable);
        }
        break;
      default:
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cameraController!.dispose();
    super.dispose();
  }
}

Future<File> getFile(String fileName) async {
  final appDir = await getTemporaryDirectory();
  final appPath = appDir.path;
  final fileOnDevice = File('$appPath/$fileName');
  await fileOnDevice.parent.create(recursive: true);
  final rawAssetFile = await rootBundle.load(fileName);
  final rawBytes = rawAssetFile.buffer.asUint8List();
  await fileOnDevice.writeAsBytes(rawBytes, flush: true);
  return fileOnDevice;
}
