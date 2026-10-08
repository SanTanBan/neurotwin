# Synthesise narration to a WAV file with the Windows OneCore (WinRT) speech engine, which offers voices the
# classic SAPI engine cannot use, e.g. "Microsoft Heera" (English, India, female).
# Usage: powershell -File tools/tts_onecore.ps1 -Text "hello" -Out out.wav [-Voice "Microsoft Heera"] [-Rate 1.0]
param(
  [Parameter(Mandatory = $true)][string]$Text,
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Voice = "Microsoft Heera",
  [double]$Rate = 1.0
)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media.SpeechSynthesis, ContentType = WindowsRuntime]
$null = [Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType = WindowsRuntime]

function Await($op, [Type]$type) {
  $m = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
  $task = $m.MakeGenericMethod($type).Invoke($null, @($op))
  $task.Wait()
  $task.Result
}

$s = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer
$v = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | Where-Object { $_.DisplayName -eq $Voice } | Select-Object -First 1
if ($v) { $s.Voice = $v } else { Write-Warning "voice '$Voice' unavailable, using default" }
$s.Options.SpeakingRate = $Rate
$stream = Await ($s.SynthesizeTextToStreamAsync($Text)) ([Windows.Media.SpeechSynthesis.SpeechSynthesisStream])
$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0))
$n = [uint32]$stream.Size
$null = Await ($reader.LoadAsync($n)) ([uint32])
$bytes = New-Object byte[] $n
$reader.ReadBytes($bytes)
[System.IO.File]::WriteAllBytes($Out, $bytes)
