# Synthesise narration to a WAV file with the Windows speech engine.
# Usage: powershell -File tools/tts.ps1 -Text "hello" -Out out.wav [-Voice "Microsoft Zira Desktop"] [-Rate 0]
param(
  [Parameter(Mandatory = $true)][string]$Text,
  [Parameter(Mandatory = $true)][string]$Out,
  [string]$Voice = "Microsoft Zira Desktop",
  [int]$Rate = 0
)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $s.SelectVoice($Voice) } catch { Write-Warning "voice '$Voice' unavailable, using default" }
$s.Rate = $Rate
$s.SetOutputToWaveFile($Out)
$s.Speak($Text)
$s.Dispose()
