package com.kokoro.android

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.kokoro.android.engine.KokoroTTSEngine
import com.kokoro.android.engine.ModelDownloader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "KokoroMainActivity"
        private const val PERMISSION_REQUEST_CODE = 100
    }

    private lateinit var etText: EditText
    private lateinit var spinnerVoice: Spinner
    private lateinit var seekBarSpeed: SeekBar
    private lateinit var tvSpeedValue: TextView
    private lateinit var btnGenerate: Button
    private lateinit var btnPlay: Button
    private lateinit var btnStop: Button
    private lateinit var tvStatus: TextView
    private lateinit var tvModelStatus: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var tvDownloadProgress: TextView

    private lateinit var ttsEngine: KokoroTTSEngine
    private lateinit var modelDownloader: ModelDownloader

    private var audioTrack: AudioTrack? = null
    private var lastAudio: FloatArray? = null
    private var isPlaying = false
    private var isDownloading = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        modelDownloader = ModelDownloader(this)
        ttsEngine = KokoroTTSEngine(this)

        initViews()
        setupListeners()
        checkPermissions()
    }

    private fun initViews() {
        etText = findViewById(R.id.etText)
        spinnerVoice = findViewById(R.id.spinnerVoice)
        seekBarSpeed = findViewById(R.id.seekBarSpeed)
        tvSpeedValue = findViewById(R.id.tvSpeedValue)
        btnGenerate = findViewById(R.id.btnGenerate)
        btnPlay = findViewById(R.id.btnPlay)
        btnStop = findViewById(R.id.btnStop)
        tvStatus = findViewById(R.id.tvStatus)
        tvModelStatus = findViewById(R.id.tvModelStatus)
        progressBar = findViewById(R.id.progressBar)
        tvDownloadProgress = findViewById(R.id.tvDownloadProgress)

        etText.setText("Hello! This is Kokoro TTS running on Android.")
        btnGenerate.isEnabled = false
        btnPlay.isEnabled = false
    }

    private fun setupListeners() {
        seekBarSpeed.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                val speed = progress / 100f
                tvSpeedValue.text = "%.1fx".format(speed)
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        btnGenerate.setOnClickListener {
            val text = etText.text.toString().trim()
            if (text.isEmpty()) {
                Toast.makeText(this, "Please enter text", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            generateSpeech(text)
        }

        btnPlay.setOnClickListener { playAudio() }
        btnStop.setOnClickListener { stopAudio() }
    }

    private fun checkPermissions() {
        val permissions = arrayOf(
            Manifest.permission.INTERNET,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.READ_EXTERNAL_STORAGE
        )

        val permissionsToRequest = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (permissionsToRequest.isNotEmpty()) {
            ActivityCompat.requestPermissions(
                this,
                permissionsToRequest.toTypedArray(),
                PERMISSION_REQUEST_CODE
            )
        } else {
            initializeApp()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            initializeApp()
        }
    }

    private fun initializeApp() {
        lifecycleScope.launch {
            if (modelDownloader.areModelsDownloaded()) {
                initTTSEngine()
            } else {
                showDownloadDialog()
            }
        }
    }

    private fun showDownloadDialog() {
        AlertDialog.Builder(this)
            .setTitle("Download Model Files")
            .setMessage("This app needs to download TTS model files (~380MB total).\n\n" +
                       "• kokoro-v1.0.onnx (~80MB)\n" +
                       "• voices-v1.0.bin (~300MB)\n\n" +
                       "Download now?")
            .setPositiveButton("Download") { _, _ -> startDownload() }
            .setNegativeButton("Exit") { _, _ -> finish() }
            .setCancelable(false)
            .show()
    }

    private fun startDownload() {
        isDownloading = true
        progressBar.visibility = View.VISIBLE
        tvDownloadProgress.visibility = View.VISIBLE
        tvStatus.text = "Downloading model files..."
        btnGenerate.isEnabled = false

        lifecycleScope.launch {
            try {
                modelDownloader.downloadModels(object : ModelDownloader.DownloadProgressCallback {
                    override fun onProgress(fileName: String, bytesRead: Long, totalBytes: Long) {
                        runOnUiThread {
                            val percent = if (totalBytes > 0) (bytesRead * 100 / totalBytes) else 0
                            tvDownloadProgress.text = "$fileName: $percent% (${formatSize(bytesRead)}/${formatSize(totalBytes)})"
                            progressBar.progress = percent.toInt()
                        }
                    }

                    override fun onComplete(fileName: String, file: File) {
                        Log.d(TAG, "Downloaded: $fileName")
                    }

                    override fun onError(fileName: String, error: String) {
                        Log.e(TAG, "Download error: $fileName - $error")
                    }
                })

                isDownloading = false
                tvDownloadProgress.visibility = View.GONE
                initTTSEngine()

            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)
                isDownloading = false
                tvDownloadProgress.visibility = View.GONE
                progressBar.visibility = View.GONE
                tvStatus.text = "Download failed: ${e.message}"
                Toast.makeText(this@MainActivity, "Download failed. Please check your internet connection.", Toast.LENGTH_LONG).show()
            }
        }
    }

    private suspend fun initTTSEngine() {
        withContext(Dispatchers.Main) {
            tvStatus.text = "Initializing TTS engine..."
            progressBar.visibility = View.VISIBLE
        }

        try {
            val success = withContext(Dispatchers.IO) {
                ttsEngine.init(
                    modelDownloader.getModelFile().absolutePath,
                    modelDownloader.getVoicesFile().absolutePath
                )
            }

            withContext(Dispatchers.Main) {
                if (success) {
                    val voices = ttsEngine.getVoices()
                    val adapter = ArrayAdapter(
                        this@MainActivity,
                        android.R.layout.simple_spinner_item,
                        voices
                    )
                    adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                    spinnerVoice.adapter = adapter

                    val defaultIndex = voices.indexOf("af_sarah")
                    if (defaultIndex >= 0) spinnerVoice.setSelection(defaultIndex)

                    val phonemizeStatus = if (ttsEngine.isPhonemizationAvailable()) {
                        "Phonemization: ON"
                    } else {
                        "Phonemization: OFF (use pre-phonemized text)"
                    }

                    tvModelStatus.text = "Model: Loaded (${voices.size} voices) | $phonemizeStatus"
                    tvStatus.text = "Ready"
                    btnGenerate.isEnabled = true
                    progressBar.visibility = View.GONE
                } else {
                    tvModelStatus.text = "Model: Failed to load"
                    tvStatus.text = "Initialization error"
                    progressBar.visibility = View.GONE
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Init failed", e)
            withContext(Dispatchers.Main) {
                tvModelStatus.text = "Model: Error"
                tvStatus.text = "Error: ${e.message}"
                progressBar.visibility = View.GONE
            }
        }
    }

    private fun generateSpeech(text: String) {
        if (!ttsEngine.isInitialized()) {
            Toast.makeText(this, "TTS engine not ready", Toast.LENGTH_SHORT).show()
            return
        }

        val voice = spinnerVoice.selectedItem?.toString() ?: "af_sarah"
        val speed = seekBarSpeed.progress / 100f

        lifecycleScope.launch {
            btnGenerate.isEnabled = false
            tvStatus.text = "Generating..."
            progressBar.visibility = View.VISIBLE

            try {
                val (audio, sampleRate) = withContext(Dispatchers.Default) {
                    ttsEngine.synthesize(
                        text = text,
                        voice = voice,
                        speed = speed,
                        isPhonemes = false  // Auto-phonemize
                    )
                }

                lastAudio = audio
                val duration = audio.size.toFloat() / sampleRate

                withContext(Dispatchers.Main) {
                    tvStatus.text = "Generated: %.2fs audio".format(duration)
                    btnPlay.isEnabled = true
                    progressBar.visibility = View.GONE
                    playAudio()
                }

            } catch (e: Exception) {
                Log.e(TAG, "Generation failed", e)
                withContext(Dispatchers.Main) {
                    tvStatus.text = "Error: ${e.message}"
                    progressBar.visibility = View.GONE
                }
            } finally {
                withContext(Dispatchers.Main) {
                    btnGenerate.isEnabled = true
                }
            }
        }
    }

    private fun playAudio() {
        val audio = lastAudio
        if (audio == null || audio.isEmpty()) {
            Toast.makeText(this, "No audio to play", Toast.LENGTH_SHORT).show()
            return
        }

        stopAudio()

        val sampleRate = KokoroConfig.SAMPLE_RATE
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        audioTrack = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(bufferSize, audio.size * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()

        val pcmData = WavWriter.toPcm16(audio)
        audioTrack?.write(pcmData, 0, pcmData.size)
        audioTrack?.play()
        isPlaying = true
        tvStatus.text = "Playing..."

        lifecycleScope.launch {
            while (isPlaying && audioTrack?.playState == AudioTrack.PLAYSTATE_PLAYING) {
                kotlinx.coroutines.delay(100)
            }
            if (isPlaying) {
                isPlaying = false
                tvStatus.text = "Playback complete"
            }
        }
    }

    private fun stopAudio() {
        isPlaying = false
        audioTrack?.stop()
        audioTrack?.release()
        audioTrack = null
    }

    private fun formatSize(bytes: Long): String {
        return when {
            bytes >= 1_000_000 -> "%.1f MB".format(bytes / 1_000_000.0)
            bytes >= 1_000 -> "%.1f KB".format(bytes / 1_000.0)
            else -> "$bytes B"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopAudio()
        ttsEngine.release()
    }
}
