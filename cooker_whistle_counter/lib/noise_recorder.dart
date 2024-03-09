import 'dart:async';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:noise_meter/noise_meter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:serious_python/serious_python.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'main.dart';
class AudioRecorderOnNoise {
  final double upperThreshold;
  final double lowerThreshold;
  bool _isRecording = false;
  late NoiseMeter _noiseMeter;
  late FlutterSoundRecorder _soundRecorder;
  late StreamSubscription<NoiseReading> _noiseSubscription;
  String _filePath = '';

  AudioRecorderOnNoise({required this.upperThreshold, required this.lowerThreshold}) {
    _noiseMeter = NoiseMeter();
    _soundRecorder = FlutterSoundRecorder();
  }

  Future<void> _requestPermissions() async {
    await Permission.microphone.request();
    await Permission.storage.request();
  }

  Future<void> init() async {
    await _requestPermissions();
    await _soundRecorder.openRecorder();
  }

  void startMonitoring() async {
    final directory = await getApplicationDocumentsDirectory();
    _noiseSubscription = _noiseMeter.noise.listen((noiseReading) async {
      if (!_isRecording && noiseReading.meanDecibel > upperThreshold) {
        await _startRecording();
      } else if (_isRecording && noiseReading.meanDecibel < lowerThreshold) {
        await _stopRecording();
        SeriousPython.run("app/app.zip", environmentVariables: {"AUDIO_DATA_PATH": '${directory.path}/audio_latest_whistle.wav'});

        var response = await http.post(
            Uri.parse("http://127.0.0.1:55001/python"),
            headers: {'Content-Type': 'application/json'},
            body: json.encode({"command"})
        );

        if (response.statusCode == 200) {
          var data = json.decode(response.body);
          if (data['prediction'] != null && data['prediction'] > 70) {
            onData(data['prediction']);
          }
        } else {
          // Handle the error case
          print('Request failed with status: ${response.statusCode}.');
        }
      }
    });
  }


  onData(var data){

  }

  Future<void> _startRecording() async {
    final directory =  await getApplicationDocumentsDirectory();
    _filePath = '${directory.path}/audio_latest_whistle.wav'; // Set file path
    await _soundRecorder.startRecorder(toFile: _filePath);
    _isRecording = true;
  }

  Future<void> _stopRecording() async {
    await _soundRecorder.stopRecorder();
    _isRecording = false;
    // Here you can handle the recorded file as needed
  }

  Future<void> dispose() async {
    await _noiseSubscription.cancel();
    await _soundRecorder.closeRecorder();
  }
}
