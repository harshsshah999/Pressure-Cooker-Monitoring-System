import 'package:flutter/material.dart';
import 'package:serious_python/serious_python.dart';
import 'package:noise_meter/noise_meter.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:async';
import 'package:flutter_ringtone_player/flutter_ringtone_player.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

void main() async{
  //runApp(const MyApp());
  runApp(NoiseMeterApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a blue toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}
class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}
class _MyHomePageState extends State<MyHomePage> {
  int _counter = 0;

  void _incrementCounter() {
    setState(() {
      SeriousPython.run("app/app.zip", environmentVariables: {"a": "1", "b": "2"});
      // This call to setState tells the Flutter framework that something has
      // changed in this State, which causes it to rerun the build method below
      // so that the display can reflect the updated values. If we changed
      // _counter without calling setState(), then the build method would not be
      // called again, and so nothing would appear to happen.
      _counter++;
    });
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Text(widget.title),
      ),
      body: Center(
        // Center is a layout widget. It takes a single child and positions it
        // in the middle of the parent.
        child: Column(
          // Column is also a layout widget. It takes a list of children and
          // arranges them vertically. By default, it sizes itself to fit its
          // children horizontally, and tries to be as tall as its parent.
          //
          // Column has various properties to control how it sizes itself and
          // how it positions its children. Here we use mainAxisAlignment to
          // center the children vertically; the main axis here is the vertical
          // axis because Columns are vertical (the cross axis would be
          // horizontal).
          //
          // TRY THIS: Invoke "debug painting" (choose the "Toggle Debug Paint"
          // action in the IDE, or press "p" in the console), to see the
          // wireframe for each widget.
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            const Text(
              'You have pushed the button this many times:',
            ),
            Text(
              '$_counter',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _incrementCounter,
        tooltip: 'Increment',
        child: const Icon(Icons.add),
      ), // This trailing comma makes auto-formatting nicer for build methods.
    );
  }
}

class NoiseMeterApp extends StatefulWidget {
  @override
  _NoiseMeterAppState createState() => _NoiseMeterAppState();
}

Future<void> requestMicrophonePermission() async {
  var status = await Permission.microphone.request();
  if (status.isGranted) {
    // Microphone permission granted
  } else {
    // Microphone permission denied
  }
}
class _NoiseMeterAppState extends State<NoiseMeterApp> {
  bool _isRecording = false;
  bool _awaitingDropBelowThreshold = false;
  NoiseReading? _latestReading;
  StreamSubscription<NoiseReading>? _noiseSubscription;
  NoiseMeter? noiseMeter;
  List<double> _lastTenSeconds = [];
  double _dropBelowThreshold = 50.0; // New variable for drop below threshold
  double _whistleThreshold = 80.0;    // New variable for whistle detection threshold
  int _sampleSize = 10;               // New variable for sample size
  int _targetWhistles = 0;
  int _detectedWhistles = 0;
  int _countAboveThresholdRequired = 7;
  TextEditingController _targetController = TextEditingController();
  FlutterLocalNotificationsPlugin flutterLocalNotificationsPlugin = FlutterLocalNotificationsPlugin();

  @override
  void initState() {
    super.initState();
    var initializationSettingsAndroid = AndroidInitializationSettings('app_icon');
    var initializationSettings = InitializationSettings(
      android: initializationSettingsAndroid,
    );
    flutterLocalNotificationsPlugin.initialize(initializationSettings);
  }

  @override
  void dispose() {
    _noiseSubscription?.cancel();
    _targetController.dispose();
    super.dispose();
  }

  void onData(NoiseReading noiseReading) {
    if (!mounted) return;
    setState(() {
      _latestReading = noiseReading;
      _lastTenSeconds.add(noiseReading.meanDecibel);
      if (_lastTenSeconds.length > _sampleSize) {
        _lastTenSeconds.removeAt(0);
      }

      // Check if we are currently waiting for the decibels to drop below 50
      if (_awaitingDropBelowThreshold) {
        if (noiseReading.meanDecibel < _dropBelowThreshold) {
          _awaitingDropBelowThreshold = false;
        }
      } else {
        int countAboveThreshold = _lastTenSeconds.where((level) => level > _whistleThreshold).length;
        if (countAboveThreshold >= _countAboveThresholdRequired) {
          _detectedWhistles++;
          if (_detectedWhistles >= _targetWhistles) {
            // Trigger the notification
            if (_isRecording) {
              FlutterRingtonePlayer.playAlarm();
              stop(); // Stop monitoring after reaching the target
            }
          }
          // Set the state to wait for the decibels to drop below 50
          _awaitingDropBelowThreshold = true;
        }
      }
    });
  }




  Future<void> _showNotification() async {
    var androidPlatformChannelSpecifics = AndroidNotificationDetails(
      'your channel id',
      'your channel name',
      importance: Importance.max,
      priority: Priority.high,
      showWhen: false,
    );
    var platformChannelSpecifics = NotificationDetails(
      android: androidPlatformChannelSpecifics,
    );
    await flutterLocalNotificationsPlugin.show(
      0,
      'Whistle Detected',
      'You have reached your target of $_targetWhistles whistles!',
      platformChannelSpecifics,
      payload: 'Item x',
    );
  }
  void onError(Object error) {
    print(error);
    stop();
  }

  Future<void> start() async {
    noiseMeter ??= NoiseMeter();
    if (!(await Permission.microphone.isGranted)) await Permission.microphone.request();
    if (!(await Permission.notification.isGranted)) await Permission.notification.request();

    _noiseSubscription = noiseMeter?.noise.listen(onData, onError: onError);
    setState(() {
      _isRecording = true;
      _lastTenSeconds.clear();
      _detectedWhistles = 0;
    });
  }

  void stop() {
    _noiseSubscription?.cancel();
    setState(() => _isRecording = false);
  }
  void stopAlarm() {
    FlutterRingtonePlayer.stop();
  }
  @override
  Widget build(BuildContext context) => MaterialApp(
    home: Scaffold(
      appBar: AppBar(
        title: Text('Cooker Whistle Detector'),
        backgroundColor: Colors.deepPurple,
      ),
      body: SingleChildScrollView( // Added for scrollability
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Target Whistles Input
              TextField(
                controller: _targetController,
                keyboardType: TextInputType.number,
                decoration: InputDecoration(
                  border: OutlineInputBorder(),
                  labelText: 'Set Target Whistles',
                  prefixIcon: Icon(Icons.settings_input_antenna),
                ),
              ),
              SizedBox(height: 10), // Spacing

              // Set Target Button
              ElevatedButton(
                onPressed: () {
                  setState(() {
                    _targetWhistles = int.tryParse(_targetController.text) ?? 0;
                  });
                },
                child: Text('Set Target'),
                style: ElevatedButton.styleFrom(
                  primary: Colors.deepPurple,
                  onPrimary: Colors.white,
                ),
              ),
              SizedBox(height: 10), // Spacing

              // Start/Stop Monitoring Button
              ElevatedButton(
                onPressed: _isRecording ? stop : start,
                child: Text(_isRecording ? 'Stop Monitoring' : 'Start Monitoring'),
                style: ElevatedButton.styleFrom(
                  primary: _isRecording ? Colors.red : Colors.green,
                ),
              ),
              SizedBox(height: 10), // Spacing

              // Stop Alarm Button
              ElevatedButton(
                onPressed: stopAlarm,
                child: Text('Stop Alarm'),
                style: ElevatedButton.styleFrom(
                  primary: Colors.blue,
                ),
              ),
              SizedBox(height: 10), // Spacing

              // Monitoring Information
              if (_isRecording) ...[
                Text(
                  'Monitoring... Detected Whistles: $_detectedWhistles',
                  style: TextStyle(fontSize: 20, color: Colors.blue),
                ),
                Text(
                  'Noise: ${_latestReading?.meanDecibel.toStringAsFixed(2)} dB',
                ),
                Text(
                  'Max: ${_latestReading?.maxDecibel.toStringAsFixed(2)} dB',
                ),
              ],

              // Sliders and Labels
              _buildSlider(
                title: 'Drop Below Threshold',
                value: _dropBelowThreshold,
                min: 1,
                max: 150,
                onChanged: (value) => setState(() => _dropBelowThreshold = value),
              ),
              _buildSlider(
                title: 'Whistle Detection Threshold',
                value: _whistleThreshold,
                min: 1,
                max: 150,
                onChanged: (value) => setState(() => _whistleThreshold = value),
              ),
              _buildSlider(
                title: 'Sample Size',
                value: _sampleSize.toDouble(),
                min: 2,
                max: 20,
                onChanged: (value) => setState(() => _sampleSize = value.toInt()),
              ),
              _buildSlider(
                title: 'Count Above Threshold Required',
                value: _countAboveThresholdRequired.toDouble(),
                min: 1,
                max: _sampleSize.toDouble(),
                onChanged: (value) => setState(() => _countAboveThresholdRequired = value.toInt()),
              ),
            ],
          ),
        ),
      ),
    ),
  );

  Widget _buildSlider({
    required String title,
    required double value,
    required double min,
    required double max,
    required Function(double) onChanged,
  }) {
    return Column(
      children: [
        Text('$title: ${value.round()}'),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: (max - min).toInt(),
          label: value.round().toString(),
          onChanged: onChanged,
        ),
        SizedBox(height: 10), // Spacing
      ],
    );
  }
}