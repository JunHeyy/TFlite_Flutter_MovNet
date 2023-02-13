/// Bundles different elapsed times
class Stats {
  /// Total time taken in the isolate where the inference runs
  late int totalPredictTime;

  /// [totalPredictTime] + communication overhead time
  /// between main isolate and another isolate
  late int totalElapsedTime;

  /// Time for which inference runs
  late int inferenceTime;

  /// Time taken to pre-process the image
  late int preProcessingTime;

  Stats(
      {required this.totalPredictTime,
      required this.totalElapsedTime,
      required this.inferenceTime,
      required this.preProcessingTime});

  @override
  String toString() {
    return 'Stats{totalPredictTime: $totalPredictTime, totalElapsedTime: $totalElapsedTime, inferenceTime: $inferenceTime, preProcessingTime: $preProcessingTime}';
  }
}
