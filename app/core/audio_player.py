"""
Audio player core — wraps QMediaPlayer for synced playback with transcription.
"""

from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput


class AudioPlayer(QObject):
    """
    Audio player that supports play/pause/seek with position tracking.
    Designed to sync with transcription segment highlighting.
    """

    position_changed = Signal(float)    # current position in seconds
    state_changed = Signal(str)         # "playing", "paused", "stopped"
    duration_changed = Signal(float)    # total duration in seconds
    error_occurred = Signal(str)        # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.8)

        self._filepath: str = ""
        self._loaded: bool = False
        self._duration_s: float = 0.0
        self._seeking: bool = False

        # Wire internal signals
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_state_changed)
        self._player.errorOccurred.connect(self._on_error)

    def load(self, filepath: str):
        """Load an audio/video file for playback."""
        if filepath == self._filepath and self._loaded:
            return
        self._filepath = filepath
        self._loaded = False
        self._player.setSource(QUrl.fromLocalFile(filepath))

    def play(self):
        """Start or resume playback."""
        if self._filepath:
            self._player.play()

    def pause(self):
        """Pause playback."""
        self._player.pause()

    def stop(self):
        """Stop playback and reset position."""
        self._player.stop()

    def seek(self, seconds: float):
        """Seek to a specific position in seconds."""
        self._seeking = True
        self._player.setPosition(int(seconds * 1000))
        self._seeking = False

    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.pause()
        else:
            self.play()

    def seek_relative(self, delta_seconds: float):
        """Seek relative to current position."""
        current = self.position
        new_pos = max(0.0, min(self._duration_s, current + delta_seconds))
        self.seek(new_pos)

    @property
    def position(self) -> float:
        """Current playback position in seconds."""
        return self._player.position() / 1000.0

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        return self._duration_s

    @property
    def state(self) -> str:
        """Current playback state: 'playing', 'paused', or 'stopped'."""
        s = self._player.playbackState()
        if s == QMediaPlayer.PlaybackState.PlayingState:
            return "playing"
        elif s == QMediaPlayer.PlaybackState.PausedState:
            return "paused"
        return "stopped"

    @property
    def is_playing(self) -> bool:
        return self.state == "playing"

    def set_rate(self, rate: float):
        """Velocidade de reprodução (0.5x–2x), preservando o tom quando o backend permite."""
        self._player.setPlaybackRate(max(0.25, min(3.0, float(rate))))

    @property
    def rate(self) -> float:
        return self._player.playbackRate()

    @property
    def filepath(self) -> str:
        return self._filepath

    def unload(self):
        """Libera o arquivo atual (ex.: origem de um item do histórico ausente)."""
        self._player.stop()
        self._player.setSource(QUrl())
        self._filepath = ""
        self._loaded = False
        self._duration_s = 0.0

    @property
    def volume(self) -> float:
        return self._audio_output.volume()

    @volume.setter
    def volume(self, value: float):
        self._audio_output.setVolume(max(0.0, min(1.0, value)))

    # --- Internal signal handlers ---

    def _on_position_changed(self, position_ms: int):
        """Convert ms to seconds and emit."""
        if not self._seeking:
            self.position_changed.emit(position_ms / 1000.0)

    def _on_duration_changed(self, duration_ms: int):
        self._duration_s = duration_ms / 1000.0
        self._loaded = True
        self.duration_changed.emit(self._duration_s)

    def _on_state_changed(self, state):
        self.state_changed.emit(self.state)

    def _on_error(self, error, message: str = ""):
        if error != QMediaPlayer.Error.NoError:
            self.error_occurred.emit(f"Audio error: {message or error}")
