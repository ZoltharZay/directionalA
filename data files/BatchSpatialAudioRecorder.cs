using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

public class BatchSpatialAudioRecorder : MonoBehaviour
{
    [Header("Folders")]
    [Tooltip("Absolute path, or relative to project root. Use forward slashes.")]
    public string inputFolderPath = "C:/temp/input_sfx";
    [Tooltip("Absolute path, or relative to project root. Use forward slashes.")]
    public string outputFolderPath = "C:/temp/output_wav";

    [Header("Targets (processed in this exact order)")]
    public List<AudioSource> staticTargets = new List<AudioSource>();
    public List<AudioSource> movingTargets = new List<AudioSource>();

    [Header("Playback Settings")]
    public int staticRepeats = 3;
    public int movingRepeats = 2;

    [Tooltip("Extra tail time added to recording after playback stops (seconds).")]
    public float recordTailSeconds = 0.25f;

    [Tooltip("Small pre-roll delay before starting playback (seconds).")]
    public float prerollSeconds = 0.05f;

    [Tooltip("Run automatically on Play.")]
    public bool autoStart = false;

    [Header("Input filtering")]
    [Tooltip("If true: ONLY .wav. If false: wav/ogg/mp3")]
    public bool wavOnly = true;

    [Header("Recording")]
    [Tooltip("Assign the scene AudioListener here (recommended). If empty, will FindObjectOfType.")]
    public AudioListener listener;

    private ListenerRecorder _recorder;

    [Serializable]
    public class DirectionSound
    {
        public AudioSource source;
    }

    private Vector3 originalLocalPos;

    private void Awake()
    {
        if (listener == null) listener = FindObjectOfType<AudioListener>();
        if (listener == null)
        {
            Debug.LogError("No AudioListener found. Add one to your scene and assign it.");
            enabled = false;
            return;
        }

        _recorder = listener.GetComponent<ListenerRecorder>();
        if (_recorder == null) _recorder = listener.gameObject.AddComponent<ListenerRecorder>();
    }

    private void Start()
    {
        if (autoStart)
            StartCoroutine(RunBatch());
    }

    [ContextMenu("Run Batch")]
    public void RunBatchFromContextMenu()
    {
        StartCoroutine(RunBatch());
    }

    private IEnumerator RunBatch()
    {
        StopAllTargets();

        string inPath = NormalizePath(inputFolderPath);
        string outPath = NormalizePath(outputFolderPath);

        if (!Directory.Exists(inPath))
        {
            Debug.LogError($"Input folder does not exist: {inPath}");
            yield break;
        }
        Directory.CreateDirectory(outPath);

        // 1) Collect ONLY audio files (wav-only OR wav/ogg/mp3)
        List<string> audioFiles = CollectAudioFiles(inPath, wavOnly);

        if (audioFiles.Count == 0)
        {
            Debug.LogError($"No supported audio files found in: {inPath} (wavOnly={wavOnly})");
            yield break;
        }

        Debug.Log($"Found {audioFiles.Count} audio files. Processing TARGET-BY-TARGET...");

        // 2) TARGET-BY-TARGET processing:
        //    For target 1: run all files
        //    Then target 2: run all files, etc.

        // --- STATIC TARGETS first, in inspector order ---
        for (int t = 0; t < staticTargets.Count; t++)
        {
            var src = staticTargets[t];
            if (src == null) continue;

            Debug.Log($"[Static Target {t + 1}/{staticTargets.Count}] {src.gameObject.name}");

            for (int i = 0; i < audioFiles.Count; i++)
            {
                int fileIndex = i + 1;
                string filePath = audioFiles[i];

                AudioClip clip = null;
                yield return StartCoroutine(LoadClip(filePath, c => clip = c));

                if (clip == null)
                {
                    Debug.LogWarning($"Skipping (failed to load): {filePath}");
                    continue;
                }

                string safeName = SanitizeFileName(src.gameObject.name);
                string outFile = Path.Combine(outPath, $"{safeName}-{fileIndex}.wav");

                yield return StartCoroutine(RecordStaticTarget(src, clip, staticRepeats, outFile));

                Destroy(clip);
                yield return null;
            }
        }

        // --- MOVING TARGETS next, in inspector order ---
        for (int t = 0; t < movingTargets.Count; t++)
        {
            var src = movingTargets[t];
            if (src == null) continue;

            Debug.Log($"[Moving Target {t + 1}/{movingTargets.Count}] {src.gameObject.name}");

            for (int i = 0; i < audioFiles.Count; i++)
            {
                int fileIndex = i + 1;
                string filePath = audioFiles[i];

                AudioClip clip = null;
                yield return StartCoroutine(LoadClip(filePath, c => clip = c));

                if (clip == null)
                {
                    Debug.LogWarning($"Skipping (failed to load): {filePath}");
                    continue;
                }

                string safeName = SanitizeFileName(src.gameObject.name);
                string outFile = Path.Combine(outPath, $"{safeName}-{fileIndex}.wav");

                yield return StartCoroutine(RecordMovingTarget(src, clip, movingRepeats, outFile));

                Destroy(clip);
                yield return null;
            }
        }

        Debug.Log("Batch complete.");
    }

    private static List<string> CollectAudioFiles(string folder, bool wavOnly)
    {
        var files = Directory.GetFiles(folder);
        Array.Sort(files, StringComparer.OrdinalIgnoreCase);

        var results = new List<string>(files.Length);
        foreach (var f in files)
        {
            string ext = Path.GetExtension(f).ToLowerInvariant();

            if (wavOnly)
            {
                if (ext == ".wav") results.Add(f);
            }
            else
            {
                if (ext == ".wav" || ext == ".ogg" || ext == ".mp3") results.Add(f);
            }
        }
        return results;
    }

    private IEnumerator RecordStaticTarget(AudioSource src, AudioClip clip, int repeats, string outFile)
    {
        StopAllTargets();

        src.clip = clip;
        src.loop = false;

        _recorder.Begin();
        yield return new WaitForSeconds(prerollSeconds);

        for (int r = 0; r < repeats; r++)
        {
            src.Play();
            yield return new WaitForSeconds(clip.length);
        }

        src.Stop();
        yield return new WaitForSeconds(recordTailSeconds);

        var samples = _recorder.End(out int sampleRate, out int channels);
        WriteWav(outFile, samples, sampleRate, channels);

        Debug.Log($"Saved: {outFile}");
    }

    private IEnumerator RecordMovingTarget(AudioSource src, AudioClip clip, int repeats, string outFile)
    {
        StopAllTargets();

        src.clip = clip;
        src.loop = false;

        _recorder.Begin();
        yield return new WaitForSeconds(prerollSeconds);

        DirectionSound ds = new DirectionSound { source = src };

        for (int r = 0; r < repeats; r++)
        {
            yield return StartCoroutine(PlayMovingSound(ds));
            yield return new WaitForSeconds(0.05f);
        }

        yield return new WaitForSeconds(recordTailSeconds);

        var samples = _recorder.End(out int sampleRate, out int channels);
        WriteWav(outFile, samples, sampleRate, channels);

        Debug.Log($"Saved: {outFile}");
    }

    private void StopAllTargets()
    {
        foreach (var s in staticTargets) if (s != null) s.Stop();
        foreach (var s in movingTargets) if (s != null) s.Stop();
    }

    // Moving logic 
private IEnumerator PlayMovingSound(DirectionSound target)
{
    AudioSource src = target.source;
    if (src == null || src.transform.childCount < 2) yield break;

    Vector3 startPos = src.transform.GetChild(0).position;
    Vector3 endPos   = src.transform.GetChild(1).position;

    src.transform.position = startPos;

 
    bool prevLoop = src.loop;
    src.loop = true;     // keep playing while moving
    src.time = 0f;       


    src.Play();

    int pingPongs = 2;
    float moveDuration = 2f;

    for (int i = 0; i < pingPongs * 2; i++)
    {
        Vector3 from = (i % 2 == 0) ? startPos : endPos;
        Vector3 to   = (i % 2 == 0) ? endPos   : startPos;

        float elapsed = 0f;
        while (elapsed < moveDuration)
        {
            src.transform.position = Vector3.Lerp(from, to, elapsed / moveDuration);
            elapsed += Time.deltaTime;
            yield return null;
        }


        src.transform.position = to;
    }

    // stop after movement is done
    src.Stop();
    src.loop = prevLoop;          // restore previous setting
    src.transform.position = startPos;
}
    // Load clip from disk
    private IEnumerator LoadClip(string path, Action<AudioClip> onLoaded)
    {
        onLoaded?.Invoke(null);

        string url = "file:///" + path.Replace("\\", "/");

        AudioType type = AudioType.UNKNOWN;
        string ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext == ".wav") type = AudioType.WAV;
        else if (ext == ".ogg") type = AudioType.OGGVORBIS;
        else if (ext == ".mp3") type = AudioType.MPEG;

        using (UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(url, type))
        {
            yield return www.SendWebRequest();

#if UNITY_2020_2_OR_NEWER
            if (www.result != UnityWebRequest.Result.Success)
#else
            if (www.isNetworkError || www.isHttpError)
#endif
            {
                Debug.LogWarning($"Failed to load clip: {path}\n{www.error}");
                yield break;
            }

            AudioClip clip = DownloadHandlerAudioClip.GetContent(www);
            onLoaded?.Invoke(clip);
        }
    }

    // Listener recording (final mix)
    private class ListenerRecorder : MonoBehaviour
    {
        private readonly object _lock = new object();
        private bool _recording = false;
        private readonly List<float> _buffer = new List<float>(48000 * 10);
        private int _channels = 2;
        private int _sampleRate = 48000;

        public void Begin()
        {
            lock (_lock)
            {
                _buffer.Clear();
                _sampleRate = AudioSettings.outputSampleRate;
                _recording = true;
            }
        }

        public float[] End(out int sampleRate, out int channels)
        {
            lock (_lock)
            {
                _recording = false;
                sampleRate = _sampleRate;
                channels = _channels;
                return _buffer.ToArray();
            }
        }

        private void OnAudioFilterRead(float[] data, int channels)
        {
            if (!_recording) return;

            lock (_lock)
            {
                _channels = channels;
                _buffer.AddRange(data);
            }
        }
    }

    // WAV writing (16-bit PCM)
    private static void WriteWav(string filePath, float[] samples, int sampleRate, int channels)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(filePath));

        short[] intData = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            float v = Mathf.Clamp(samples[i], -1f, 1f);
            intData[i] = (short)Mathf.RoundToInt(v * short.MaxValue);
        }

        byte[] bytes = new byte[intData.Length * 2];
        Buffer.BlockCopy(intData, 0, bytes, 0, bytes.Length);

        using (FileStream fs = new FileStream(filePath, FileMode.Create, FileAccess.Write))
        using (BinaryWriter bw = new BinaryWriter(fs))
        {
            int byteRate = sampleRate * channels * 2;
            int subChunk2Size = bytes.Length;
            int chunkSize = 36 + subChunk2Size;

            bw.Write(System.Text.Encoding.ASCII.GetBytes("RIFF"));
            bw.Write(chunkSize);
            bw.Write(System.Text.Encoding.ASCII.GetBytes("WAVE"));

            bw.Write(System.Text.Encoding.ASCII.GetBytes("fmt "));
            bw.Write(16);
            bw.Write((short)1);
            bw.Write((short)channels);
            bw.Write(sampleRate);
            bw.Write(byteRate);
            bw.Write((short)(channels * 2));
            bw.Write((short)16);

            bw.Write(System.Text.Encoding.ASCII.GetBytes("data"));
            bw.Write(subChunk2Size);
            bw.Write(bytes);
        }
    }

    private static string NormalizePath(string p)
    {
        if (string.IsNullOrWhiteSpace(p)) return p;

        if (!Path.IsPathRooted(p))
        {
            string projectRoot = Directory.GetParent(Application.dataPath).FullName;
            p = Path.Combine(projectRoot, p);
        }
        return p.Replace("\\", "/");
    }

    private static string SanitizeFileName(string name)
    {
        foreach (char c in Path.GetInvalidFileNameChars())
            name = name.Replace(c, '_');
        return name;
    }
}
