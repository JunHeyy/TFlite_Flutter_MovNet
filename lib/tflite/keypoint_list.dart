import 'dart:math';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:object_detection/tflite/keypoint.dart';

////// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// // /// /// /// ///
///    [KeypointRecognition] holds a set of Keypoints for the current frame      ///
///                 For eg, nose, left_eye, right_eye                           ///
/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// ///

const KEYPOINT_DICT = [
  'nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle'
];

const TOTAL_KEYPOINTS = 17;

/// Represents the recognition output from the model
class KeypointRecognition {
  // Index of the result

  List<Keypoint>? listOfKeypointLocationInOneFrame;

  List<Keypoint>? get getKeypointLocationinOneFrame =>
      listOfKeypointLocationInOneFrame;

  KeypointRecognition({required this.listOfKeypointLocationInOneFrame});
  /*
  @override
  String toString() {
    String keypointsString = '';
    int counter = 0;
    //for (Keypoint keypoint in listOfKeypointLocationInOneFrame) {
    //  print('${KEYPOINT_DICT[counter]} : ${keypoint.toString()}');
    //  counter += 1;
    //}
    return keypointsString;
  }*/
}
