import org.gradle.api.tasks.Exec
import org.gradle.api.tasks.bundling.AbstractArchiveTask

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "uk.megalith.lidar"
    compileSdk = 35

    defaultConfig {
        applicationId = "uk.megalith.lidar"
        minSdk = 26
        targetSdk = 35
        versionCode = 320
        versionName = "3.2.0"
    }

    // Omit AGP's Play-specific DependencyInfoBlock so F-Droid scanner and
    // reproducible-build verification pass without extra signing blocks.
    dependenciesInfo {
        includeInApk = false
        includeInBundle = false
    }

    signingConfigs {
        val envKsPath = System.getenv("KEYSTORE_PATH") ?: System.getenv("RELEASE_KEYSTORE_PATH")
        val ksFile = if (!envKsPath.isNullOrBlank()) file(envKsPath) else rootProject.file("megalith-lidar-release.jks")
        if (ksFile.exists()) {
            create("release") {
                storeFile = ksFile
                storePassword = System.getenv("KEYSTORE_PASSWORD") ?: System.getenv("RELEASE_KEYSTORE_PASSWORD") ?: ""
                keyAlias = System.getenv("KEY_ALIAS") ?: System.getenv("RELEASE_KEY_ALIAS") ?: ""
                keyPassword = System.getenv("KEY_PASSWORD") ?: System.getenv("RELEASE_KEY_PASSWORD") ?: ""
                enableV1Signing = false
                enableV2Signing = true
                enableV3Signing = true
            }
        }
    }

    buildTypes {
        release {
            signingConfig = signingConfigs.findByName("release")
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            vcsInfo.include = false
        }
        debug {
            // Sign debug with the release cert when the keystore env is present
            // (source ~/.megalith-lidar-release.env), so a debuggable build installs over
            // a release-signed device build without a data-wiping uninstall.
            // Falls back to the default debug keystore when the env is absent.
            if (signingConfigs.findByName("release") != null &&
                (System.getenv("KEYSTORE_PASSWORD") != null || System.getenv("RELEASE_KEYSTORE_PASSWORD") != null)
            ) {
                signingConfig = signingConfigs.getByName("release")
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs {
            useLegacyPackaging = false
        }
    }

    lint {
        if (project.hasProperty("noLintVital")) {
            checkReleaseBuilds = false
        }
        abortOnError = false
    }
}

val prepareWebAssets by tasks.registering(Exec::class) {
    val rootRepoDir = rootProject.projectDir.parentFile
    workingDir = rootRepoDir
    inputs.file(File(rootRepoDir, "index.html"))
    inputs.file(File(rootRepoDir, "scripts/prepare_android_assets.py"))
    inputs.file(File(rootRepoDir, "scripts/export_osmand_plugin.py"))
    inputs.dir(file("src/main/leaflet"))
    inputs.file(File(rootRepoDir, "docs/survey_summary.json"))
    inputs.file(File(rootRepoDir, "docs/candidates.geojson"))
    inputs.file(File(rootRepoDir, "docs/roman_candidates.geojson"))
    inputs.file(File(rootRepoDir, "docs/enclosure_candidates.geojson"))
    outputs.dir(file("src/main/assets/web"))

    commandLine("python3", File(rootRepoDir, "scripts/prepare_android_assets.py").absolutePath)
}

tasks.named("preBuild") {
    dependsOn(prepareWebAssets)
}

tasks.withType<AbstractArchiveTask>().configureEach {
    isPreserveFileTimestamps = false
    isReproducibleFileOrder = true
}

dependencies {
    implementation("androidx.webkit:webkit:1.12.1")
}
