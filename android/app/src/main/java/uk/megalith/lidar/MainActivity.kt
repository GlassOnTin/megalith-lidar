package uk.megalith.lidar

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.ActivityNotFoundException
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.view.WindowInsets
import android.webkit.GeolocationPermissions
import android.webkit.JavascriptInterface
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.webkit.WebViewAssetLoader
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

class MainActivity : Activity() {

    private lateinit var rootLayout: FrameLayout
    private lateinit var webView: WebView
    private var pendingGeoOrigin: String? = null
    private var pendingGeoCallback: GeolocationPermissions.Callback? = null
    private var fileUploadCallback: ValueCallback<Array<Uri>>? = null

    inner class AndroidBridge {
        @JavascriptInterface
        fun saveOrShareFile(filename: String, mimeType: String, content: String) {
            runOnUiThread {
                try {
                    val safeName = filename.replace(Regex("[^A-Za-z0-9._-]"), "_")
                        .ifEmpty { "megalith_export.txt" }
                    val exportsDir = File(cacheDir, "exports").apply { mkdirs() }
                    val outFile = File(exportsDir, safeName)
                    outFile.writeText(content, Charsets.UTF_8)
                    shareExportedFile(outFile, mimeType)
                } catch (e: Exception) {
                    Toast.makeText(
                        this@MainActivity,
                        "Export failed: ${e.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        window.statusBarColor = Color.parseColor("#1F2E28")
        window.navigationBarColor = Color.parseColor("#1F2E28")

        rootLayout = FrameLayout(this).apply {
            setBackgroundColor(Color.parseColor("#1F2E28"))
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            rootLayout.setOnApplyWindowInsetsListener { view, insets ->
                val bars = insets.getInsets(
                    WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout()
                )
                view.setPadding(bars.left, bars.top, bars.right, bars.bottom)
                insets
            }
        }

        webView = WebView(this).apply {
            setBackgroundColor(Color.parseColor("#F2ECE1"))
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        rootLayout.addView(webView)
        setContentView(rootLayout)

        val assetLoader = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()

        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            databaseEnabled = true
            setGeolocationEnabled(true)
            allowFileAccess = false
            allowContentAccess = false
            mediaPlaybackRequiresUserGesture = false
            cacheMode = WebSettings.LOAD_DEFAULT
            builtInZoomControls = false
            displayZoomControls = false
        }

        webView.addJavascriptInterface(AndroidBridge(), "MegalithAndroid")

        webView.webViewClient = object : WebViewClient() {
            override fun shouldInterceptRequest(
                view: WebView,
                request: WebResourceRequest
            ): WebResourceResponse? {
                val url = request.url
                val path = url.path ?: ""
                if (url.host == APP_HOST && path.startsWith("/assets/web/docs/diagnostics/")) {
                    val fileName = path.removePrefix("/assets/web/docs/diagnostics/")
                    return serveDiagnosticPng(fileName)
                }
                return assetLoader.shouldInterceptRequest(url)
            }

            override fun shouldOverrideUrlLoading(
                view: WebView,
                request: WebResourceRequest
            ): Boolean {
                val uri = request.url ?: return false
                val scheme = uri.scheme?.lowercase() ?: ""
                if (scheme == "geo") {
                    openExternalUri(uri)
                    return true
                }
                if (uri.host == APP_HOST) {
                    val path = uri.path ?: ""
                    if (isDownloadableAssetPath(path)) {
                        shareBundledAsset(path.removePrefix("/assets/"))
                        return true
                    }
                    return false
                }
                if (scheme == "http" || scheme == "https") {
                    openExternalUri(uri)
                    return true
                }
                return false
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onGeolocationPermissionsShowPrompt(
                origin: String,
                callback: GeolocationPermissions.Callback
            ) {
                if (hasLocationPermission()) {
                    callback.invoke(origin, true, false)
                } else {
                    pendingGeoOrigin = origin
                    pendingGeoCallback = callback
                    requestPermissions(
                        arrayOf(
                            Manifest.permission.ACCESS_FINE_LOCATION,
                            Manifest.permission.ACCESS_COARSE_LOCATION
                        ),
                        REQ_LOCATION_PERMISSION
                    )
                }
            }

            override fun onShowFileChooser(
                webView: WebView,
                filePathCallback: ValueCallback<Array<Uri>>,
                fileChooserParams: FileChooserParams
            ): Boolean {
                fileUploadCallback?.onReceiveValue(null)
                fileUploadCallback = filePathCallback
                val intent = Intent(Intent.ACTION_GET_CONTENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                    putExtra(
                        Intent.EXTRA_MIME_TYPES,
                        arrayOf("application/json", "application/geo+json", "text/plain")
                    )
                }
                return try {
                    startActivityForResult(
                        Intent.createChooser(intent, "Select Survey Notes JSON"),
                        REQ_FILE_CHOOSER
                    )
                    true
                } catch (_: ActivityNotFoundException) {
                    fileUploadCallback = null
                    false
                }
            }
        }

        webView.setDownloadListener { url, _, _, mimeType, _ ->
            val parsed = Uri.parse(url)
            if (parsed.host == APP_HOST && parsed.path != null) {
                shareBundledAsset(parsed.path!!.removePrefix("/assets/"), mimeType)
            } else if (parsed.scheme == "http" || parsed.scheme == "https") {
                openExternalUri(parsed)
            }
        }

        if (savedInstanceState == null) {
            webView.loadUrl(START_URL)
        } else {
            webView.restoreState(savedInstanceState)
        }
    }

    private fun isDownloadableAssetPath(path: String): Boolean {
        val low = path.lowercase()
        return low.endsWith(".osf") ||
            low.endsWith(".gpx") ||
            low.endsWith(".osm") ||
            low.endsWith("candidates.geojson") ||
            low.endsWith("maproulette_os1888_antiquities.geojson") ||
            low.endsWith("survey_summary.json")
    }

    private fun shareBundledAsset(assetRelPath: String, overrideMime: String? = null) {
        try {
            val fileName = assetRelPath.substringAfterLast('/')
            val exportsDir = File(cacheDir, "exports").apply { mkdirs() }
            val outFile = File(exportsDir, fileName)
            assets.open(assetRelPath).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
            val mime = when {
                fileName.endsWith(".osf", true) -> "application/octet-stream"
                fileName.endsWith(".gpx", true) -> "application/gpx+xml"
                fileName.endsWith(".geojson", true) -> "application/geo+json"
                fileName.endsWith(".json", true) -> "application/json"
                fileName.endsWith(".osm", true) -> "application/xml"
                !overrideMime.isNullOrBlank() -> overrideMime
                else -> "application/octet-stream"
            }
            shareExportedFile(outFile, mime)
        } catch (e: Exception) {
            Toast.makeText(this, "Unable to export file: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun shareExportedFile(file: File, mimeType: String) {
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        val sendIntent = Intent(Intent.ACTION_SEND).apply {
            type = mimeType
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        val chooser = Intent.createChooser(sendIntent, file.name).apply {
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(chooser)
    }

    private fun serveDiagnosticPng(rawName: String): WebResourceResponse {
        val safeName = rawName.replace(Regex("[^A-Za-z0-9._-]"), "")
        if (safeName.isEmpty() || !safeName.endsWith(".png")) {
            return emptyResponse(404, "Not Found")
        }
        val diagDir = File(cacheDir, "diagnostics").apply { mkdirs() }
        val cachedFile = File(diagDir, safeName)
        if (!cachedFile.exists() || cachedFile.length() == 0L) {
            val remoteUrl = "$REMOTE_DIAG_BASE$safeName"
            try {
                val conn = (URL(remoteUrl).openConnection() as HttpURLConnection).apply {
                    connectTimeout = 8000
                    readTimeout = 12000
                    requestMethod = "GET"
                }
                if (conn.responseCode == 200) {
                    val tmpFile = File(diagDir, "$safeName.tmp")
                    conn.inputStream.use { input ->
                        FileOutputStream(tmpFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                    tmpFile.renameTo(cachedFile)
                }
                conn.disconnect()
            } catch (_: Exception) {
                // Offline or network unreachable; fall through to 404 if not cached
            }
        }
        if (cachedFile.exists() && cachedFile.length() > 0L) {
            return WebResourceResponse(
                "image/png",
                null,
                200,
                "OK",
                mapOf(
                    "Access-Control-Allow-Origin" to "*",
                    "Cache-Control" to "public, max-age=31536000"
                ),
                FileInputStream(cachedFile)
            )
        }
        return emptyResponse(404, "Not Found")
    }

    private fun emptyResponse(code: Int, reason: String): WebResourceResponse {
        return WebResourceResponse(
            "text/plain",
            "utf-8",
            code,
            reason,
            mapOf("Access-Control-Allow-Origin" to "*"),
            ByteArrayInputStream(ByteArray(0))
        )
    }

    private fun openExternalUri(uri: Uri) {
        try {
            val intent = Intent(Intent.ACTION_VIEW, uri).apply {
                addCategory(Intent.CATEGORY_BROWSABLE)
            }
            startActivity(intent)
        } catch (_: ActivityNotFoundException) {
            Toast.makeText(this, "No app installed to open ${uri.scheme} link", Toast.LENGTH_SHORT)
                .show()
        }
    }

    private fun hasLocationPermission(): Boolean {
        return checkSelfPermission(Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQ_LOCATION_PERMISSION) {
            val granted = grantResults.any { it == PackageManager.PERMISSION_GRANTED }
            val origin = pendingGeoOrigin
            val cb = pendingGeoCallback
            pendingGeoOrigin = null
            pendingGeoCallback = null
            if (origin != null && cb != null) {
                cb.invoke(origin, granted, false)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_FILE_CHOOSER) {
            val cb = fileUploadCallback ?: return
            fileUploadCallback = null
            if (resultCode == RESULT_OK && data?.data != null) {
                cb.onReceiveValue(arrayOf(data.data!!))
            } else {
                cb.onReceiveValue(null)
            }
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        webView.saveState(outState)
    }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        webView.evaluateJavascript(
            "(function(){var d=document.getElementById('osmandQrDialog');if(d&&d.open){d.close();return true;}return false;})()"
        ) { res ->
            if (res == "true") {
                return@evaluateJavascript
            }
            if (webView.canGoBack()) {
                webView.goBack()
            } else {
                @Suppress("DEPRECATION")
                super.onBackPressed()
            }
        }
    }

    companion object {
        private const val APP_HOST = "appassets.androidplatform.net"
        private const val START_URL =
            "https://appassets.androidplatform.net/assets/web/index.html"
        private const val REMOTE_DIAG_BASE =
            "https://raw.githubusercontent.com/GlassOnTin/megalith-lidar/main/docs/diagnostics/"
        private const val REQ_LOCATION_PERMISSION = 1001
        private const val REQ_FILE_CHOOSER = 1002
    }
}
