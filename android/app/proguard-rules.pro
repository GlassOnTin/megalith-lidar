# Keep MainActivity and the WebView JavascriptInterface bridge class + member
# names intact so tagged Log statements, stack traces, and window.MegalithAndroid
# reflective bridge calls (saveOrShareFile) survive R8 minification.
-keep class uk.megalith.lidar.MainActivity { *; }
-keep class uk.megalith.lidar.MainActivity$AndroidBridge { *; }
-keepclassmembers class uk.megalith.lidar.MainActivity$AndroidBridge {
    @android.webkit.JavascriptInterface <methods>;
}
