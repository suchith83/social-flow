# Flutter Integration Guide

This guide provides comprehensive instructions for integrating the Social Flow Backend API with your Flutter frontend application.

## 📱 Flutter Setup

### Dependencies

Add the following dependencies to your `pubspec.yaml`:

```yaml
dependencies:
  flutter:
    sdk: flutter
  
  # HTTP and API
  http: ^1.1.0
  dio: ^5.3.2
  
  # State Management
  provider: ^6.0.5
  riverpod: ^2.4.0
  flutter_riverpod: ^2.4.0
  
  # Authentication
  flutter_secure_storage: ^9.0.0
  jwt_decoder: ^2.0.1
  
  # Video and Media
  video_player: ^2.8.1
  chewie: ^1.7.4
  image_picker: ^1.0.4
  file_picker: ^6.1.1
  
  # UI Components
  cached_network_image: ^3.3.0
  shimmer: ^3.0.0
  pull_to_refresh: ^2.0.0
  
  # Real-time
  web_socket_channel: ^2.4.0
  
  # Notifications
  firebase_messaging: ^14.7.10
  flutter_local_notifications: ^16.3.0
  
  # Utilities
  intl: ^0.18.1
  shared_preferences: ^2.2.2
  connectivity_plus: ^5.0.2
  permission_handler: ^11.0.1
```

## 🔧 API Client Setup

### Base API Client

Create a base API client class:

```dart
// lib/services/api_client.dart
import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class ApiClient {
  static const String baseUrl = 'http://localhost:8000/api/v1';
  static const String _storageKey = 'auth_tokens';
  
  late Dio _dio;
  final FlutterSecureStorage _storage = const FlutterSecureStorage();
  
  ApiClient() {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 30),
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    ));
    
    _setupInterceptors();
  }
  
  void _setupInterceptors() {
    _dio.interceptors.add(
      InterceptorsWrapper(
        onRequest: (options, handler) async {
          final token = await _storage.read(key: 'access_token');
          if (token != null) {
            options.headers['Authorization'] = 'Bearer $token';
          }
          handler.next(options);
        },
        onError: (error, handler) async {
          if (error.response?.statusCode == 401) {
            await _refreshToken();
            // Retry the request
            final response = await _dio.fetch(error.requestOptions);
            handler.resolve(response);
          } else {
            handler.next(error);
          }
        },
      ),
    );
  }
  
  Future<void> _refreshToken() async {
    try {
      final refreshToken = await _storage.read(key: 'refresh_token');
      if (refreshToken != null) {
        final response = await _dio.post('/auth/refresh', data: {
          'refresh_token': refreshToken,
        });
        
        if (response.statusCode == 200) {
          final data = response.data;
          await _storage.write(key: 'access_token', value: data['access_token']);
          await _storage.write(key: 'refresh_token', value: data['refresh_token']);
        }
      }
    } catch (e) {
      // Handle refresh token failure
      await _storage.deleteAll();
      // Navigate to login screen
    }
  }
  
  // Generic HTTP methods
  Future<Response> get(String path, {Map<String, dynamic>? queryParameters}) {
    return _dio.get(path, queryParameters: queryParameters);
  }
  
  Future<Response> post(String path, {dynamic data, Map<String, dynamic>? queryParameters}) {
    return _dio.post(path, data: data, queryParameters: queryParameters);
  }
  
  Future<Response> put(String path, {dynamic data, Map<String, dynamic>? queryParameters}) {
    return _dio.put(path, data: data, queryParameters: queryParameters);
  }
  
  Future<Response> delete(String path, {Map<String, dynamic>? queryParameters}) {
    return _dio.delete(path, queryParameters: queryParameters);
  }
  
  // File upload
  Future<Response> uploadFile(
    String path,
    File file, {
    Map<String, dynamic>? data,
    ProgressCallback? onSendProgress,
  }) {
    final formData = FormData.fromMap({
      'file': MultipartFile.fromFileSync(file.path),
      ...?data,
    });
    
    return _dio.post(
      path,
      data: formData,
      onSendProgress: onSendProgress,
    );
  }
}
```

## 🔐 Authentication Service

Create an authentication service:

```dart
// lib/services/auth_service.dart
import 'dart:convert';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'api_client.dart';

class AuthService {
  final ApiClient _apiClient = ApiClient();
  final FlutterSecureStorage _storage = const FlutterSecureStorage();
  
  // User model
  static const String _userKey = 'current_user';
  
  Future<Map<String, dynamic>?> register({
    required String username,
    required String email,
    required String password,
    required String displayName,
    String? bio,
    String? avatarUrl,
    String? website,
    String? location,
  }) async {
    try {
      final response = await _apiClient.post('/auth/register', data: {
        'username': username,
        'email': email,
        'password': password,
        'display_name': displayName,
        'bio': bio,
        'avatar_url': avatarUrl,
        'website': website,
        'location': location,
      });
      
      if (response.statusCode == 201) {
        return response.data;
      }
      throw Exception('Registration failed: ${response.data}');
    } catch (e) {
      throw Exception('Registration error: $e');
    }
  }
  
  Future<Map<String, dynamic>?> login({
    required String username,
    required String password,
  }) async {
    try {
      final response = await _apiClient.post('/auth/login', data: {
        'username': username,
        'password': password,
      });
      
      if (response.statusCode == 200) {
        final data = response.data;
        await _storage.write(key: 'access_token', value: data['access_token']);
        await _storage.write(key: 'refresh_token', value: data['refresh_token']);
        return data;
      }
      throw Exception('Login failed: ${response.data}');
    } catch (e) {
      throw Exception('Login error: $e');
    }
  }
  
  Future<void> logout() async {
    await _storage.deleteAll();
  }
  
  Future<bool> isLoggedIn() async {
    final token = await _storage.read(key: 'access_token');
    return token != null;
  }
  
  Future<Map<String, dynamic>?> getCurrentUser() async {
    try {
      final response = await _apiClient.get('/users/me');
      if (response.statusCode == 200) {
        final userData = response.data;
        await _storage.write(key: _userKey, value: jsonEncode(userData));
        return userData;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  Future<Map<String, dynamic>?> getStoredUser() async {
    final userJson = await _storage.read(key: _userKey);
    if (userJson != null) {
      return jsonDecode(userJson);
    }
    return null;
  }
}
```

## 📹 Video Service

Create a video service for handling video operations:

```dart
// lib/services/video_service.dart
import 'dart:io';
import 'api_client.dart';

class VideoService {
  final ApiClient _apiClient = ApiClient();
  
  Future<Map<String, dynamic>?> uploadVideo({
    required File videoFile,
    required String title,
    String? description,
    String? tags,
    Function(double)? onProgress,
  }) async {
    try {
      final response = await _apiClient.uploadFile(
        '/videos/upload',
        videoFile,
        data: {
          'title': title,
          'description': description ?? '',
          'tags': tags ?? '',
        },
        onSendProgress: (sent, total) {
          if (onProgress != null) {
            onProgress(sent / total);
          }
        },
      );
      
      if (response.statusCode == 200) {
        return response.data;
      }
      throw Exception('Video upload failed: ${response.data}');
    } catch (e) {
      throw Exception('Video upload error: $e');
    }
  }
  
  Future<Map<String, dynamic>?> getVideo(String videoId) async {
    try {
      final response = await _apiClient.get('/videos/$videoId');
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  Future<Map<String, dynamic>?> getVideoStreamUrl(
    String videoId, {
    String quality = 'auto',
  }) async {
    try {
      final response = await _apiClient.get(
        '/videos/$videoId/stream',
        queryParameters: {'quality': quality},
      );
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  Future<bool> likeVideo(String videoId) async {
    try {
      final response = await _apiClient.post('/videos/$videoId/like');
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
  
  Future<bool> unlikeVideo(String videoId) async {
    try {
      final response = await _apiClient.delete('/videos/$videoId/like');
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
  
  Future<bool> recordView(String videoId) async {
    try {
      final response = await _apiClient.post('/videos/$videoId/view');
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
  
  Future<Map<String, dynamic>?> createLiveStream({
    required String title,
    String? description,
  }) async {
    try {
      final response = await _apiClient.post('/videos/live/create', data: {
        'title': title,
        'description': description ?? '',
      });
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  Future<List<Map<String, dynamic>>> getVideos({
    int skip = 0,
    int limit = 100,
    String? category,
  }) async {
    try {
      final response = await _apiClient.get('/videos', queryParameters: {
        'skip': skip,
        'limit': limit,
        if (category != null) 'category': category,
      });
      if (response.statusCode == 200) {
        return List<Map<String, dynamic>>.from(response.data);
      }
      return [];
    } catch (e) {
      return [];
    }
  }
}
```

## 🤖 ML Service

Create an ML service for AI features:

```dart
// lib/services/ml_service.dart
import 'dart:convert';
import 'api_client.dart';

class MLService {
  final ApiClient _apiClient = ApiClient();
  
  Future<Map<String, dynamic>?> analyzeContent({
    required String contentType,
    required Map<String, dynamic> contentData,
  }) async {
    try {
      final response = await _apiClient.post('/ml/analyze', data: {
        'content_type': contentType,
        'content_data': jsonEncode(contentData),
      });
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  Future<Map<String, dynamic>?> getRecommendations({
    String contentType = 'mixed',
    int limit = 10,
  }) async {
    try {
      final response = await _apiClient.get('/ml/recommendations', queryParameters: {
        'content_type': contentType,
        'limit': limit,
      });
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
  
  Future<Map<String, dynamic>?> getTaskStatus(String taskId) async {
    try {
      final response = await _apiClient.get('/ml/tasks/$taskId');
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
}
```

## 📊 Analytics Service

Create an analytics service:

```dart
// lib/services/analytics_service.dart
import 'api_client.dart';

class AnalyticsService {
  final ApiClient _apiClient = ApiClient();
  
  Future<bool> trackEvent({
    required String eventType,
    required Map<String, dynamic> data,
  }) async {
    try {
      final response = await _apiClient.post('/analytics/track', data: {
        'event_type': eventType,
        'data': data,
      });
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
  
  Future<Map<String, dynamic>?> getUserAnalytics({
    required String userId,
    String timePeriod = '7d',
  }) async {
    try {
      final response = await _apiClient.get(
        '/analytics/user/$userId',
        queryParameters: {'time_period': timePeriod},
      );
      if (response.statusCode == 200) {
        return response.data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
}
```

## 🎥 Video Player Widget

Create a custom video player widget:

```dart
// lib/widgets/video_player_widget.dart
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'package:chewie/chewie.dart';
import '../services/video_service.dart';

class VideoPlayerWidget extends StatefulWidget {
  final String videoId;
  final String? streamingUrl;
  final String quality;
  
  const VideoPlayerWidget({
    Key? key,
    required this.videoId,
    this.streamingUrl,
    this.quality = 'auto',
  }) : super(key: key);
  
  @override
  State<VideoPlayerWidget> createState() => _VideoPlayerWidgetState();
}

class _VideoPlayerWidgetState extends State<VideoPlayerWidget> {
  VideoPlayerController? _controller;
  ChewieController? _chewieController;
  final VideoService _videoService = VideoService();
  bool _isLoading = true;
  String? _error;
  
  @override
  void initState() {
    super.initState();
    _initializePlayer();
  }
  
  Future<void> _initializePlayer() async {
    try {
      String? streamingUrl = widget.streamingUrl;
      
      if (streamingUrl == null) {
        final streamData = await _videoService.getVideoStreamUrl(
          widget.videoId,
          quality: widget.quality,
        );
        streamingUrl = streamData?['streaming_url'];
      }
      
      if (streamingUrl != null) {
        _controller = VideoPlayerController.networkUrl(Uri.parse(streamingUrl));
        await _controller!.initialize();
        
        _chewieController = ChewieController(
          videoPlayerController: _controller!,
          autoPlay: false,
          looping: false,
          showControls: true,
          materialProgressColors: ChewieProgressColors(
            playedColor: Colors.red,
            handleColor: Colors.red,
            backgroundColor: Colors.grey,
            bufferedColor: Colors.lightGreen,
          ),
        );
        
        setState(() {
          _isLoading = false;
        });
      } else {
        setState(() {
          _error = 'Failed to load video stream';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = 'Error loading video: $e';
        _isLoading = false;
      });
    }
  }
  
  @override
  void dispose() {
    _chewieController?.dispose();
    _controller?.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    if (_isLoading) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }
    
    if (_error != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 64,
              color: Colors.red[300],
            ),
            const SizedBox(height: 16),
            Text(
              _error!,
              style: const TextStyle(color: Colors.red),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  _isLoading = true;
                  _error = null;
                });
                _initializePlayer();
              },
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }
    
    return Chewie(controller: _chewieController!);
  }
}
```

## 🔄 State Management

Create a state management setup using Riverpod:

```dart
// lib/providers/auth_provider.dart
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../services/auth_service.dart';

final authServiceProvider = Provider<AuthService>((ref) => AuthService());

final currentUserProvider = StateNotifierProvider<CurrentUserNotifier, Map<String, dynamic>?>((ref) {
  return CurrentUserNotifier(ref.read(authServiceProvider));
});

class CurrentUserNotifier extends StateNotifier<Map<String, dynamic>?> {
  final AuthService _authService;
  
  CurrentUserNotifier(this._authService) : super(null) {
    _loadCurrentUser();
  }
  
  Future<void> _loadCurrentUser() async {
    final user = await _authService.getStoredUser();
    state = user;
  }
  
  Future<void> login(String username, String password) async {
    final result = await _authService.login(username: username, password: password);
    if (result != null) {
      final user = await _authService.getCurrentUser();
      state = user;
    }
  }
  
  Future<void> logout() async {
    await _authService.logout();
    state = null;
  }
  
  Future<void> refreshUser() async {
    final user = await _authService.getCurrentUser();
    state = user;
  }
}
```

## 📱 Usage Examples

### Login Screen

```dart
// lib/screens/login_screen.dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import '../providers/auth_provider.dart';

class LoginScreen extends ConsumerStatefulWidget {
  const LoginScreen({Key? key}) : super(key: key);
  
  @override
  ConsumerState<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends ConsumerState<LoginScreen> {
  final _formKey = GlobalKey<FormState>();
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();
  bool _isLoading = false;
  
  @override
  void dispose() {
    _usernameController.dispose();
    _passwordController.dispose();
    super.dispose();
  }
  
  Future<void> _login() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
      });
      
      try {
        await ref.read(currentUserProvider.notifier).login(
          _usernameController.text,
          _passwordController.text,
        );
        
        if (mounted) {
          Navigator.of(context).pushReplacementNamed('/home');
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Login failed: $e')),
          );
        }
      } finally {
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
        }
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Login')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              TextFormField(
                controller: _usernameController,
                decoration: const InputDecoration(labelText: 'Username'),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter username';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _passwordController,
                decoration: const InputDecoration(labelText: 'Password'),
                obscureText: true,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter password';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 24),
              ElevatedButton(
                onPressed: _isLoading ? null : _login,
                child: _isLoading
                    ? const CircularProgressIndicator()
                    : const Text('Login'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
```

### Video Upload Screen

```dart
// lib/screens/video_upload_screen.dart
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import '../services/video_service.dart';

class VideoUploadScreen extends StatefulWidget {
  const VideoUploadScreen({Key? key}) : super(key: key);
  
  @override
  State<VideoUploadScreen> createState() => _VideoUploadScreenState();
}

class _VideoUploadScreenState extends State<VideoUploadScreen> {
  final _formKey = GlobalKey<FormState>();
  final _titleController = TextEditingController();
  final _descriptionController = TextEditingController();
  final _tagsController = TextEditingController();
  
  File? _selectedFile;
  bool _isUploading = false;
  double _uploadProgress = 0.0;
  
  final VideoService _videoService = VideoService();
  
  @override
  void dispose() {
    _titleController.dispose();
    _descriptionController.dispose();
    _tagsController.dispose();
    super.dispose();
  }
  
  Future<void> _pickVideo() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
    );
    
    if (result != null && result.files.isNotEmpty) {
      setState(() {
        _selectedFile = File(result.files.first.path!);
      });
    }
  }
  
  Future<void> _uploadVideo() async {
    if (_formKey.currentState!.validate() && _selectedFile != null) {
      setState(() {
        _isUploading = true;
        _uploadProgress = 0.0;
      });
      
      try {
        final result = await _videoService.uploadVideo(
          videoFile: _selectedFile!,
          title: _titleController.text,
          description: _descriptionController.text,
          tags: _tagsController.text,
          onProgress: (progress) {
            setState(() {
              _uploadProgress = progress;
            });
          },
        );
        
        if (mounted) {
          if (result != null) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Video uploaded successfully!')),
            );
            Navigator.of(context).pop();
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Video upload failed')),
            );
          }
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Upload error: $e')),
          );
        }
      } finally {
        if (mounted) {
          setState(() {
            _isUploading = false;
          });
        }
      }
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Upload Video')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            children: [
              // File picker
              GestureDetector(
                onTap: _pickVideo,
                child: Container(
                  height: 200,
                  decoration: BoxDecoration(
                    border: Border.all(color: Colors.grey),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: _selectedFile != null
                      ? Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const Icon(Icons.video_file, size: 64),
                            const SizedBox(height: 8),
                            Text(_selectedFile!.path.split('/').last),
                          ],
                        )
                      : const Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.add, size: 64),
                            Text('Tap to select video'),
                          ],
                        ),
                ),
              ),
              const SizedBox(height: 16),
              
              // Form fields
              TextFormField(
                controller: _titleController,
                decoration: const InputDecoration(labelText: 'Title'),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter title';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _descriptionController,
                decoration: const InputDecoration(labelText: 'Description'),
                maxLines: 3,
              ),
              const SizedBox(height: 16),
              TextFormField(
                controller: _tagsController,
                decoration: const InputDecoration(labelText: 'Tags (comma-separated)'),
              ),
              const SizedBox(height: 24),
              
              // Upload progress
              if (_isUploading) ...[
                LinearProgressIndicator(value: _uploadProgress),
                const SizedBox(height: 8),
                Text('Uploading... ${(_uploadProgress * 100).toInt()}%'),
                const SizedBox(height: 16),
              ],
              
              // Upload button
              ElevatedButton(
                onPressed: _isUploading ? null : _uploadVideo,
                child: _isUploading
                    ? const CircularProgressIndicator()
                    : const Text('Upload Video'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
```

## 🔧 Configuration

### Environment Configuration

Create environment-specific configurations:

```dart
// lib/config/app_config.dart
class AppConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8000/api/v1',
  );
  
  static const String environment = String.fromEnvironment(
    'ENVIRONMENT',
    defaultValue: 'development',
  );
  
  static bool get isProduction => environment == 'production';
  static bool get isDevelopment => environment == 'development';
}
```

### Build Configurations

Create different build configurations:

```yaml
# android/app/build.gradle
android {
    buildTypes {
        debug {
            buildConfigField "String", "API_BASE_URL", '"http://10.0.2.2:8000/api/v1"'
        }
        release {
            buildConfigField "String", "API_BASE_URL", '"https://api.socialflow.com/api/v1"'
        }
    }
}
```

## 🚀 Deployment

### Android

1. **Build APK**
   ```bash
   flutter build apk --release
   ```

2. **Build App Bundle**
   ```bash
   flutter build appbundle --release
   ```

### iOS

1. **Build iOS**
   ```bash
   flutter build ios --release
   ```

2. **Archive for App Store**
   ```bash
   flutter build ipa --release
   ```

## 📱 Testing

### Unit Tests

```dart
// test/services/auth_service_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/mockito.dart';
import 'package:mockito/annotations.dart';
import '../lib/services/auth_service.dart';
import '../lib/services/api_client.dart';

@GenerateMocks([ApiClient])
void main() {
  group('AuthService', () {
    late AuthService authService;
    late MockApiClient mockApiClient;
    
    setUp(() {
      mockApiClient = MockApiClient();
      authService = AuthService();
    });
    
    test('should login successfully', () async {
      // Arrange
      when(mockApiClient.post('/auth/login', data: anyNamed('data')))
          .thenAnswer((_) async => Response(
                statusCode: 200,
                data: {'access_token': 'token', 'refresh_token': 'refresh'},
              ));
      
      // Act
      final result = await authService.login(
        username: 'testuser',
        password: 'password',
      );
      
      // Assert
      expect(result, isNotNull);
      expect(result!['access_token'], equals('token'));
    });
  });
}
```

### Integration Tests

```dart
// integration_test/app_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:socialflow/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();
  
  group('App Integration Tests', () {
    testWidgets('should login and navigate to home', (tester) async {
      app.main();
      await tester.pumpAndSettle();
      
      // Test login flow
      await tester.enterText(find.byKey(const Key('username_field')), 'testuser');
      await tester.enterText(find.byKey(const Key('password_field')), 'password');
      await tester.tap(find.byKey(const Key('login_button')));
      await tester.pumpAndSettle();
      
      // Verify navigation to home
      expect(find.byKey(const Key('home_screen')), findsOneWidget);
    });
  });
}
```

## 🔒 Security Considerations

1. **Token Storage**: Use Flutter Secure Storage for sensitive data
2. **Certificate Pinning**: Implement certificate pinning for production
3. **Input Validation**: Validate all user inputs
4. **Error Handling**: Don't expose sensitive information in errors
5. **Network Security**: Use HTTPS in production

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Flutter Documentation](https://flutter.dev/docs)
- [Riverpod Documentation](https://riverpod.dev/)
- [Dio HTTP Client](https://pub.dev/packages/dio)
- [Video Player Plugin](https://pub.dev/packages/video_player)

## 🆘 Support

For Flutter integration support:
- Check the [API Documentation](http://localhost:8000/docs)
- Review the [OpenAPI Specification](openapi.yaml)
- Contact the backend team for API issues
- Check Flutter documentation for UI/UX questions
