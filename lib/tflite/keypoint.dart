import 'package:flutter/material.dart';

////// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// // /// /// /// /// /// /// /// // /// ///
///    [Keypoint] is a model class that descripts the keypoints with the properities of x,y confidence   ///
/// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// /// // // /// // // /// ///

class Keypoint {
  double x_coord;
  double y_coord;
  double confidence_score;

  Keypoint(
      {required this.x_coord,
      required this.y_coord,
      required this.confidence_score});

  @override
  String toString() {
    return "x_coord: ${this.x_coord} y_coord: ${this.y_coord} confidence_score: ${this.confidence_score}";
  }
}
